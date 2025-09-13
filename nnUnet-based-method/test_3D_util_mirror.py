import math
import os
import glob
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from torch.cuda.amp import autocast
from skimage import filters, measure
from scipy.ndimage import gaussian_filter
from networks.net_factory_3d import net_factory_3d


def save(_output_file, _case_results):
    with open(str(_output_file), "w") as f:
        json.dump(_case_results, f)


def load_clicks_data(clicks_data):
    """解析JSON格式的交互点数据（竞赛兼容，实际不使用）"""
    left_points = []
    right_points = []

    for item in clicks_data.get('points', []):
        point = item['point']  # [x, y, z]
        name = item['name']

        if name == 'Left_IAC':
            left_points.append(point)
        elif name == 'Right_IAC':
            right_points.append(point)

    print(f"Loaded {len(left_points)} Left_IAC clicks and {len(right_points)} Right_IAC clicks")
    return left_points, right_points


def process_case(case_name):
    """生成符合evalutils格式的case结果"""
    return {
        "outputs": [
            dict(type="metaio_image", filename=case_name)
        ],
        "inputs": [
            dict(type="metaio_image", filename=case_name)
        ],
        "error_messages": [],
    }


def compute_gaussian(tile_size, sigma_scale=1. / 8, dtype=np.float16):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def normalized(image):
    # 根据plan.json中的训练数据统计信息
    # 使用percentile进行clipping，更稳健
    LowerBound = -169.8032684326172  # percentile_00_5
    UpperBound = 1081.4932861328125  # percentile_99_5

    # 或者使用更保守的min/max
    # LowerBound, UpperBound = -1000.0, 3220.0

    image = np.clip(image, LowerBound, UpperBound)

    # nnUNet标准的Z-score归一化
    # 使用训练时的全局统计量（更稳定）
    train_mean = 295.36297607421875
    train_std = 195.0432586669922

    image = (image.astype(np.float32) - train_mean) / train_std

    return image


def resample_sitk_to_spacing(itk_img, out_spacing=(0.3, 0.3, 0.3),
                             is_label=False, force_size=None):
    """
    重采样到指定 spacing。
    如果 force_size 不为 None，则强制输出尺寸一致。
    """
    original_spacing = np.array(itk_img.GetSpacing(), dtype=np.float64)
    original_size = np.array(itk_img.GetSize(), dtype=np.int64)

    out_spacing = np.array(out_spacing, dtype=np.float64)
    if force_size is not None:
        # 转成 python list[int]，避免 VectorUInt32 报错
        out_size = [int(x) for x in force_size]
    else:
        out_size = np.round(original_size * (original_spacing / out_spacing)).astype(np.int64).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(out_spacing.tolist()))
    resampler.SetSize(out_size)
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    resampler.SetOutputDirection(itk_img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    # 影像用线性插值，标签用最近邻插值
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

    return resampler.Execute(itk_img)


def normalized_v2(image):
    mean = image.mean()
    std = image.std()
    image = (image - mean) / std
    return image.astype(np.float32)


def test_single_case_v2(save_model_list, model_weights, image, patch_size, num_classes=1, do_mirroring=False,
                        use_gaussian=True):
    print(f"using TTA: {do_mirroring}")
    print("Accelerated version.")
    w, h, d = image.shape
    # image = normalized_v2(image)
    with autocast():
        with torch.no_grad():
            # if the size of image is less than patch_size, then padding it
            add_pad = False
            if w < patch_size[0]:
                w_pad = patch_size[0] - w
                add_pad = True
            else:
                w_pad = 0
            if h < patch_size[1]:
                h_pad = patch_size[1] - h
                add_pad = True
            else:
                h_pad = 0
            if d < patch_size[2]:
                d_pad = patch_size[2] - d
                add_pad = True
            else:
                d_pad = 0
            wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
            hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
            dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
            if add_pad:
                image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                       (dl_pad, dr_pad)], mode='constant', constant_values=0)
            ww, hh, dd = image.shape

            step_size = 0.5
            target_step_sizes_in_voxels = [i * step_size for i in patch_size]
            sx = math.ceil((ww - patch_size[0]) / target_step_sizes_in_voxels[0]) + 1
            sy = math.ceil((hh - patch_size[1]) / target_step_sizes_in_voxels[1]) + 1
            sz = math.ceil((dd - patch_size[2]) / target_step_sizes_in_voxels[2]) + 1
            print("{}, {}, {}".format(sx, sy, sz))

            num_steps = [sx, sy, sz]
            steps = []
            for dim in range(len(patch_size)):
                # the highest step value for this dimension is
                max_step_value = image.shape[dim] - patch_size[dim]
                if num_steps[dim] > 1:
                    actual_step_size = max_step_value / (num_steps[dim] - 1)
                else:
                    actual_step_size = 99999999999  # does not matter because there is only one step at 0

                steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
                steps.append(steps_here)
            # score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
            # cnt = np.zeros(image.shape).astype(np.float32)
            score_map_torch = torch.zeros((num_classes,) + image.shape, dtype=torch.float16).cuda()
            cnt_torch = torch.zeros(image.shape, dtype=torch.float16).cuda()
            # print(score_map_torch.shape, score_map_torch.dtype)

            if use_gaussian:
                gaussian = compute_gaussian(patch_size)
                # make sure nothing is rounded to zero or we get division by zero :-(
                mn = gaussian.min()
                if mn == 0:
                    gaussian.clip(min=mn)
                gaussian_torch = torch.from_numpy(gaussian).cuda()

            image_torch = torch.from_numpy(image.astype(np.float16)).cuda()

            for j, save_model_path in enumerate(save_model_list):
                net = net_factory_3d(net_type='nnUNetv2', in_chns=1, class_num=3).cuda()
                # net.load_state_dict(torch.load(save_model_path)['network_weights'])
                state = torch.load(save_model_path, weights_only=False)  # 允许完整反序列化
                net.load_state_dict(state['network_weights'])

                net.eval()
                # print(f"load from {save_model_path}")
                for x in range(0, sx):
                    xs = steps[0][x]
                    for y in range(0, sy):
                        ys = steps[1][y]
                        for z in range(0, sz):
                            zs = steps[2][z]
                            test_patch = image_torch[xs:xs + patch_size[0],
                                         ys:ys + patch_size[1], zs:zs + patch_size[2]]
                            test_patch = test_patch[None, None, :, :, :]

                            y = net(test_patch)[0]
                            # y = torch.softmax(y, dim=1, dtype=torch.float16)
                            y = y[0, :, :, :, :]

                            if use_gaussian:
                                score_map_torch[:, xs:xs + patch_size[0], ys:ys + patch_size[1],
                                zs:zs + patch_size[2]] += y * gaussian_torch
                                cnt_torch[xs:xs + patch_size[0], ys:ys + patch_size[1],
                                zs:zs + patch_size[2]] += gaussian_torch
                            else:
                                score_map_torch[:, xs:xs + patch_size[0], ys:ys + patch_size[1],
                                zs:zs + patch_size[2]] += y
                                cnt_torch[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1

            # 在GPU上完成所有计算，避免大数组传输
            score_map_torch = score_map_torch / cnt_torch.unsqueeze(0)
            label_map_torch = torch.argmax(score_map_torch, dim=0)

            # 只传输最终的小结果到CPU
            label_map = label_map_torch.cpu().data.numpy().astype(np.uint8)

            print(f"Final result shape: {label_map.shape}, dtype: {label_map.dtype}")

            # 清理GPU内存
            del score_map_torch, cnt_torch, label_map_torch
            torch.cuda.empty_cache()

            if add_pad:
                label_map = label_map[wl_pad:wl_pad + w,
                            hl_pad:hl_pad + h, dl_pad:dl_pad + d]

    return label_map.astype(np.uint8)


def remove_small_connected_object(npy_mask, area_least=10):
    from skimage import measure
    from skimage.morphology import label

    npy_mask[npy_mask != 0] = 1
    labeled_mask, num = label(npy_mask, return_num=True)
    print('Num of Connected Objects', num)
    if num == 2:
        print('No Postprocessing...')
        return npy_mask
    else:
        print('Postprocessing...')
        region_props = measure.regionprops(labeled_mask)

        res_mask = np.zeros_like(npy_mask)
        for i in range(1, num + 1):
            t_area = region_props[i - 1].area
            if t_area > area_least:
                res_mask[labeled_mask == i] = 1

    return res_mask


def connected_component(image):
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image

    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num + 1)]
    area_list = [region[i - 1].area for i in num_list]
    print(num_list, area_list)
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x - 1])[::-1]
    print(num_list_sorted)
    # 去除面积较小的连通域
    if len(num_list_sorted) > 2:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[2:]:
            # label[label==i] = 0
            label[region[i - 1].slice][region[i - 1].image] = 0
        # num_list_sorted = num_list_sorted[:1]
    return label


def find_clicks_json_file(image_path, input_dir="/input"):
    """查找对应的点击JSON文件"""
    filename = Path(image_path).name

    # 从 ToothFairy3F_001_0.nii.gz 提取基础名称和步骤
    if filename.endswith(".nii.gz"):
        base = filename[:-7]  # remove '.nii.gz'
    elif filename.endswith(".mha"):
        base = filename[:-4]  # remove '.mha'
    else:
        base = filename

    # 构造可能的JSON文件名
    json_candidates = [
        f"{input_dir}/iac_clicks_{base}.json",
        f"{input_dir}/iac_clicks_{filename}",
    ]

    # 检查哪个文件存在
    for json_path in json_candidates:
        if os.path.isfile(json_path):
            return json_path

    # 如果找不到精确匹配，查找包含"clicks"的JSON文件
    json_files = [f for f in glob.glob(f"{input_dir}/*.json") if "clicks" in f]
    if len(json_files) == 1:
        print(f"Using single JSON file found: {json_files[0]}")
        return json_files[0]

    print(f"Warning: No clicks JSON file found for {filename}")
    return None


def test_all_case_without_score(save_model_list, model_weights, model_name, base_dir, num_classes=3,
                                patch_size=(80, 160, 192), json_path=None, test_save_path=None, TTA_flag=False):
    print("Testing begin - Competition Interactive Mode")
    path = os.listdir(base_dir)
    _case_results = []

    # 使用plans.json中3d_fullres的spacing
    target_spacing = (0.30000001192092896, 0.30000001192092896, 0.30000001192092896)

    # 缓存已处理的基础病例，避免重复推理
    case_cache = {}

    for image_path in path:
        print('Processing', image_path)

        # 1) 查找并读取对应的点击JSON文件
        clicks_json_path = find_clicks_json_file(image_path, "/input")
        if clicks_json_path:
            try:
                with open(clicks_json_path, 'r') as f:
                    clicks_data = json.load(f)
                left_points, right_points = load_clicks_data(clicks_data)
            except Exception as e:
                print(f"Warning: Failed to load clicks JSON {clicks_json_path}: {e}")
                left_points, right_points = [], []
        else:
            left_points, right_points = [], []

        # 2) 确定基础病例名称（用于缓存）
        filename = Path(image_path).name
        if filename.endswith('.nii.gz'):
            base_case = filename[:-7]  # remove .nii.gz
        elif filename.endswith('.mha'):
            base_case = filename[:-4]  # remove .mha
        else:
            base_case = filename

        # 提取基础病例名（去除步骤后缀_0, _1等）
        base_case_root = base_case.rsplit('_', 1)[0] if '_' in base_case else base_case

        # 3) 检查缓存，避免重复推理
        if base_case_root not in case_cache:
            print(f"Performing nnUNet inference for base case: {base_case_root}")

            # 读取原始图像，保存几何信息
            itk_img = sitk.ReadImage(os.path.join(base_dir, image_path))

            # 重采样到训练时的spacing
            itk_img_rs = resample_sitk_to_spacing(itk_img, out_spacing=target_spacing, is_label=False)
            img_np = sitk.GetArrayFromImage(itk_img_rs).astype(np.float32)

            # 归一化
            img_np = normalized(img_np)

            # 预测（这里是真正的nnUNet推理）
            prediction_all = test_single_case_v2(
                save_model_list, model_weights, img_np, patch_size, num_classes=3, do_mirroring=TTA_flag)

            # 后处理
            prediction_all = postprocessing(prediction_all)

            # 转换为ITK图像，使用重采样后的几何信息
            pred_itk_rs = sitk.GetImageFromArray(prediction_all.astype(np.uint8))
            pred_itk_rs.CopyInformation(itk_img_rs)

            # 重采样回原图spacing和size
            pred_itk_final = resample_sitk_to_spacing(
                pred_itk_rs,
                out_spacing=itk_img.GetSpacing(),
                is_label=True,
                force_size=itk_img.GetSize()  # 强制与原图尺寸一致
            )
            pred_itk_final.CopyInformation(itk_img)  # 确保几何信息完全匹配

            # 缓存结果
            case_cache[base_case_root] = pred_itk_final
            print(f"Cached result for {base_case_root}")
        else:
            print(f"Using cached result for {base_case_root}")
            pred_itk_final = case_cache[base_case_root]

        # 4) 保存结果（每个交互步骤都使用相同的分割结果）
        if image_path.endswith('.nii.gz'):
            output_filename = f"{base_case_root}.nii.gz"
        elif image_path.endswith('.mha'):
            output_filename = f"{base_case_root}.mha"
        else:
            # 默认输出.nii.gz格式
            output_filename = f"{base_case_root}.nii.gz"
        output_filename = image_path
        output_path = os.path.join(test_save_path, output_filename)
        sitk.WriteImage(pred_itk_final, output_path)

        print(f"Saved: {output_path}")
        print(f"Simulated interactive step with {len(left_points + right_points)} total clicks")

        # 5) 记录结果
        _case_results.append(process_case(output_filename))

    # 保存结果JSON
    if json_path:
        save(json_path, _case_results)

    print(f"Testing complete. Processed {len(path)} files, cached {len(case_cache)} unique cases.")
    return "Testing end"


def postprocessing(prediction):
    print("postprocessing for mandibular nerve segmentation")

    # 不需要标签映射，直接进行连通域后处理
    output_array = np.uint8(np.zeros_like(prediction))

    for label in [1, 2]:  # 左右下颌神经
        if label not in np.unique(prediction):
            continue

        # 提取当前标签的mask
        pred_array_label = np.zeros_like(prediction)
        pred_array_label[prediction == label] = 1

        # 连通域分析
        itk_mask = sitk.GetImageFromArray(pred_array_label)
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)
        output_mask = cc_filter.Execute(itk_mask)
        lss_filter = sitk.LabelShapeStatisticsImageFilter()
        lss_filter.Execute(output_mask)
        num_connected_label = cc_filter.GetObjectCount()
        np_output_mask = sitk.GetArrayFromImage(output_mask)

        # 使用500的面积阈值
        area_thresh = 500
        print(f"Label {label}, area_thresh: {area_thresh}")

        res_mask = np.zeros_like(np_output_mask)
        for i in range(1, num_connected_label + 1):
            area = lss_filter.GetNumberOfPixels(i)
            if area > area_thresh:
                res_mask[np_output_mask == i] = 1
        output_array[res_mask == 1] = label

    return output_array


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
               (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    return np.array([dice])