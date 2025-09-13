from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import json
import glob
import os
from typing import Dict, Tuple

from evalutils import SegmentationAlgorithm
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


def get_default_device():
    """Set device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_clicks_data(clicks_data):
    """解析JSON格式的交互点数据"""
    left_points = []
    right_points = []

    for item in clicks_data.get('points', []):
        point = item['point']  # [x, y, z]
        name = item['name']

        if name == 'Left_IAC':
            left_points.append(point)
        elif name == 'Right_IAC':
            right_points.append(point)

    return left_points, right_points


def your_oral_pharyngeal_segmentation_algorithm(input_tensor: torch.Tensor, clicks_data: Dict,
                                                session: nnInteractiveInferenceSession) -> np.ndarray:
    """
    nnInteractive based segmentation algorithm

    Args:
        input_tensor: Preprocessed CBCT volume tensor [1, H, W, D]
        clicks_data: Dictionary containing interaction points
        session: Initialized nnInteractive session

    Returns:
        Segmentation mask as numpy array
    """
    # Remove batch dimension for processing
    volume = input_tensor.squeeze(0)  # Remove batch dimension: [H, W, D]

    # Parse clicks data
    left_points, right_points = load_clicks_data(clicks_data)

    print(f"Received {len(left_points)} Left_IAC clicks and {len(right_points)} Right_IAC clicks")

    # If no points, return empty segmentation
    if len(left_points) == 0 and len(right_points) == 0:
        return np.zeros_like(volume.cpu().numpy(), dtype=np.uint8)

    # Prepare image for nnInteractive (add batch dimension back)
    img = input_tensor.cpu().numpy()  # [1, H, W, D]
    session.set_image(img)

    # Initialize combined result
    combined_result = torch.zeros(volume.shape, dtype=torch.uint8)

    # Process Left IAC if points exist
    if len(left_points) > 0:
        target_tensor_left = torch.zeros(volume.shape, dtype=torch.uint8)
        session.set_target_buffer(target_tensor_left)
        session.reset_interactions()

        # Add all left IAC points
        for point in left_points:
            x, y, z = point
            session.add_point_interaction((x, y, z), include_interaction=True)

        # Get left segmentation result
        left_result = session.target_buffer.clone()
        combined_result[left_result > 0] = 1  # Left IAC = label 1

    # Process Right IAC if points exist
    if len(right_points) > 0:
        target_tensor_right = torch.zeros(volume.shape, dtype=torch.uint8)
        session.set_target_buffer(target_tensor_right)
        session.reset_interactions()

        # Add all right IAC points
        for point in right_points:
            x, y, z = point
            session.add_point_interaction((x, y, z), include_interaction=True)

        # Get right segmentation result
        right_result = session.target_buffer.clone()
        combined_result[right_result > 0] = 2  # Right IAC = label 2

    return combined_result.numpy().astype(np.uint8)


class ToothFairy3_OralPharyngealSegmentation(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            input_path=Path('/input/images/cbct/'),
            output_path=Path('/output/images/iac-segmentation/'),
            validators={},
        )

        # Create output directory if it doesn't exist
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)

        # Create metadata output directory
        self.metadata_output_path = Path('/output/metadata/')
        if not self.metadata_output_path.exists():
            self.metadata_output_path.mkdir(parents=True)

        # Initialize device
        self.device = get_default_device()
        print(f"Using device: {self.device}")

        # Initialize nnInteractive session
        self.session = self._initialize_nninteractive()

    def _initialize_nninteractive(self):
        """Initialize nnInteractive session"""
        print("Initializing nnInteractive session...")

        # Model path in Docker container
        MODEL_PATH = "/opt/app/models/nnInteractive/nnInteractive_v1.0"

        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"nnInteractive model not found at: {MODEL_PATH}")

        session = nnInteractiveInferenceSession(
            device=self.device,
            use_torch_compile=False,
            verbose=False,
            torch_n_threads=os.cpu_count(),
            do_autozoom=True,
            use_pinned_memory=True if self.device.type == "cuda" else False,
        )

        # Load the trained model
        session.initialize_from_trained_model_folder(MODEL_PATH)
        print("nnInteractive model loaded successfully!")

        return session

    def save_instance_metadata(self, metadata: Dict, image_name: str):
        """
        Save instance metadata to JSON file

        Args:
            metadata: Instance metadata dictionary
            image_name: Name of the input image (without extension)
        """
        metadata_file = self.metadata_output_path / f"{image_name}_instances.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Segment IAC using nnInteractive
        segmented_ = self.predict(input_image=input_image, input_image_file_path=input_image_file_path)

        # Write resulting segmentation to output location
        segmentation_path = self._output_path / input_image_file_path.name
        if not self._output_path.exists():
            self._output_path.mkdir()
        sitk.WriteImage(segmented_, str(segmentation_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image, input_image_file_path: str = None) -> sitk.Image:
        input_array = sitk.GetArrayFromImage(input_image)

        # === Load and parse the JSON clicks file ===
        filename = Path(input_image_file_path).name

        if filename.endswith(".nii.gz"):
            base = filename[:-7]  # remove '.nii.gz' (7 chars)
        elif filename.endswith(".mha"):
            base = filename[:-4]  # remove '.mha' (4 chars)
        else:
            raise ValueError("Unsupported file extension")

        parts = base.split('_')
        input_json_clicks = f"/input/iac_clicks_{parts[0]}_{parts[-1]}.json"
        if not os.path.isfile(input_json_clicks):
            input_json_clicks = f"/input/iac_clicks_{base}.json"
        if not os.path.isfile(input_json_clicks):
            # Look for exactly one JSON file in /input/ that has the keyword "clicks"
            json_files = [f for f in glob.glob("/input/*.json") if "clicks" in f]
            print(json_files)
            if len(json_files) == 1:
                input_json_clicks = json_files[0]
                print(f"Using single JSON file found: {input_json_clicks}")
            else:
                raise RuntimeError(f"Could not find clicks JSON file at '{input_json_clicks}', "
                                   f"and found {len(json_files)} JSON files in /input/: {json_files}")

        try:
            with open(input_json_clicks, 'r') as f:
                clicks_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON clicks file '{input_json_clicks}': {e}")

        # Basic preprocessing (keep nnInteractive's expected preprocessing minimal)
        input_array = input_array.astype(np.float32)

        input_tensor = torch.from_numpy(input_array)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        output_array = your_oral_pharyngeal_segmentation_algorithm(input_tensor, clicks_data, self.session)

        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)

        return output_image


if __name__ == "__main__":
    ToothFairy3_OralPharyngealSegmentation().process()