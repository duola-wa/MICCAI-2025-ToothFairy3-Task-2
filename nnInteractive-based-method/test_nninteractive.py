import os
import json
import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


def load_clicks_data(json_path):
    """解析你的JSON格式的交互点数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    left_points = []
    right_points = []

    for item in data.get('points', []):
        point = item['point']  # [x, y, z]
        name = item['name']

        if name == 'Left_IAC':
            left_points.append(point)
        elif name == 'Right_IAC':
            right_points.append(point)

    return left_points, right_points


def test_single_case(session, cbct_path, json_path, output_dir):
    """测试单个案例"""

    # 1. 加载CBCT图像
    input_image = sitk.ReadImage(str(cbct_path))
    img = sitk.GetArrayFromImage(input_image)[None]  # 添加batch维度: (1, z, y, x)

    if img.ndim != 4:
        raise ValueError("Input image must be 4D with shape (1, z, y, x)")

    # 2. 设置图像到session
    session.set_image(img)

    # 3. 加载交互点
    left_points, right_points = load_clicks_data(json_path)

    # 4. 分割左侧IAC
    target_tensor_left = torch.zeros(img.shape[1:], dtype=torch.uint8)
    session.set_target_buffer(target_tensor_left)
    session.reset_interactions()

    # 添加左侧IAC的所有点
    for point in left_points:
        x, y, z = point
        session.add_point_interaction((x, y, z), include_interaction=True)

    # 获取左侧分割结果
    left_result = session.target_buffer.clone()

    # 5. 分割右侧IAC
    target_tensor_right = torch.zeros(img.shape[1:], dtype=torch.uint8)
    session.set_target_buffer(target_tensor_right)
    session.reset_interactions()

    # 添加右侧IAC的所有点
    for point in right_points:
        x, y, z = point
        session.add_point_interaction((x, y, z), include_interaction=True)

    # 获取右侧分割结果
    right_result = session.target_buffer.clone()

    # 6. 合并左右分割结果
    combined_result = torch.zeros_like(left_result, dtype=torch.uint8)
    combined_result[left_result > 0] = 1  # 左侧IAC标签为1
    combined_result[right_result > 0] = 2  # 右侧IAC标签为2

    # 7. 保存结果
    output_path = output_dir / f"{cbct_path.stem}_segmentation.nii.gz"

    # 转换回SimpleITK格式并保存
    result_array = combined_result.numpy()
    result_image = sitk.GetImageFromArray(result_array)
    result_image.CopyInformation(input_image)  # 复制原图像的空间信息

    sitk.WriteImage(result_image, str(output_path))

    return output_path


def main():
    # 配置路径
    DATA_DIR = Path("./data")
    IMAGES_DIR = DATA_DIR / "imagesTr"
    CLICKS_DIR = DATA_DIR / "ToothFairy3_clicks"
    OUTPUT_DIR = Path("./results")

    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 模型路径配置
    MODEL_PATH = "./models/nnInteractive/nnInteractive_v1.0"  # 你的实际模型路径

    # 1. 检查本地模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"本地模型不存在: {MODEL_PATH}")
        print("请检查模型路径是否正确")
        return
    else:
        print(f"使用本地模型: {MODEL_PATH}")

    # 2. 初始化推理会话
    print("初始化nnInteractive推理会话...")

    # 检查可用GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU:")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

        # 使用第一个GPU（通常显示较少的那个）
        device = torch.device("cuda:0")
        print(f"使用设备: {device}")
    else:
        device = torch.device("cpu")
        print("未检测到CUDA，使用CPU")

    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,  # RTX 3090完全支持，但先保持关闭确保稳定性
        verbose=False,  # 关闭详细输出
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,  # 自动缩放，对大图像很有用
        use_pinned_memory=True,  # RTX 3090的大显存可以充分利用
    )

    # 3. 加载训练好的模型
    session.initialize_from_trained_model_folder(MODEL_PATH)
    print("模型加载完成!")

    # 4. 获取所有数据文件
    cbct_files = sorted(list(IMAGES_DIR.glob("ToothFairy3*_*_0000.nii.gz")))
    print(f"找到 {len(cbct_files)} 个CBCT文件")

    if len(cbct_files) == 0:
        print("错误: 未找到CBCT文件，请检查路径是否正确")
        return

    # 5. 处理所有案例
    print(f"开始处理所有 {len(cbct_files)} 个案例...")

    for i, cbct_path in enumerate(cbct_files):
        print(f"\n=== 案例 {i + 1}/{len(cbct_files)}: {cbct_path.name} ===")

        # 构建对应的JSON文件路径
        # 从 ToothFairy3X_Y_0000.nii.gz 提取 X_Y
        # 例如：ToothFairy3F_001_0000.nii.gz -> F_001
        cbct_filename = cbct_path.name  # ToothFairy3F_001_0000.nii.gz
        # 移除 .nii.gz 后缀
        base_name = cbct_filename.replace(".nii.gz", "")  # ToothFairy3F_001_0000
        # 移除 _0000 后缀
        name_without_suffix = base_name.replace("_0000", "")  # ToothFairy3F_001

        json_filename = f"{name_without_suffix}_clicks.json"  # ToothFairy3F_001_clicks.json
        json_path = CLICKS_DIR / json_filename

        if not json_path.exists():
            print(f"  ✗ JSON文件不存在: {json_filename}")
            continue

        try:
            result_path = test_single_case(session, cbct_path, json_path, OUTPUT_DIR)
            print(f"  ✓ 处理完成，保存至: {result_path.name}")

        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            continue

    print(f"\n测试完成! 结果保存在: {OUTPUT_DIR}")
    print("你可以用ITK-SNAP或其他医学图像查看器查看分割结果")


if __name__ == "__main__":
    main()