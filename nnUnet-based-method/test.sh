#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-interactive-algorithm"

echo "Testing ToothFairy3 Interactive-Segmentation algorithm..."

# Define paths - 使用你本地的数据路径
IMAGES_DIR="$SCRIPTPATH/data/imagesTr"
CLICKS_DIR="$SCRIPTPATH/data/ToothFairy3_clicks"
RESULTS_DIR="$SCRIPTPATH/test-results"
ALGORITHM_OUTPUT="$RESULTS_DIR/algorithm-output"

# Create test directories
mkdir -p "$SCRIPTPATH/test/input/images/cbct"
mkdir -p "$SCRIPTPATH/test/input"
mkdir -p "$SCRIPTPATH/test/output/images/iac-segmentation"
mkdir -p "$SCRIPTPATH/test/output/metadata"
mkdir -p "$ALGORITHM_OUTPUT"

# Copy test images and associated clicks to input directory
echo "Copying test image-click pairs from $IMAGES_DIR and $CLICKS_DIR..."

for image_file in "$IMAGES_DIR"/*.nii.gz; do
    if [ -f "$image_file" ]; then
        # 从 ToothFairy3F_001_0000.nii.gz 提取基础名称
        base_name=$(basename "$image_file" .nii.gz)
        base_name="${base_name%_0000}"  # 移除 _0000 -> ToothFairy3F_001

        echo "Processing case: $base_name"

        # 找到对应的点击文件
        click_file="$CLICKS_DIR/${base_name}_clicks.json"

        if [ -f "$click_file" ]; then
            echo "  Found click file: ${base_name}_clicks.json"

            # 模拟6个交互步骤 (0-5个点)
            for i in $(seq 0 5); do
                new_image_name="${base_name}_${i}.nii.gz"
                new_click_name="iac_clicks_${base_name}_${i}.json"

                # 复制图像文件（每个步骤都是同一个图像）
                cp "$image_file" "$SCRIPTPATH/test/input/images/cbct/$new_image_name"

                # 生成对应步骤的点击文件
                python3 - <<EOF
import json
import sys

# 读取原始点击文件
with open('$click_file', 'r') as f:
    data = json.load(f)

# 提取左右IAC的点
left_points = []
right_points = []

for item in data.get('points', []):
    if item['name'] == 'Left_IAC':
        left_points.append(item)
    elif item['name'] == 'Right_IAC':
        right_points.append(item)

# 修正的交互点分配策略
step = $i
output_points = []

if step == 0:
    # 第0步：无点击
    pass
elif step == 1:
    # 第1步：1个左侧点
    output_points = left_points[:1]
elif step == 2:
    # 第2步：1个左侧 + 1个右侧
    output_points = left_points[:1] + right_points[:1]
elif step == 3:
    # 第3步：2个左侧 + 1个右侧
    output_points = left_points[:2] + right_points[:1]
elif step == 4:
    # 第4步：2个左侧 + 2个右侧
    output_points = left_points[:2] + right_points[:2]
elif step == 5:
    # 第5步：3个左侧 + 2个右侧（或者全部点）
    if len(left_points) >= 3 and len(right_points) >= 2:
        output_points = left_points[:3] + right_points[:2]
    else:
        # 如果点数不够，使用所有可用的点
        output_points = left_points + right_points

# 保存为竞赛期望的格式
output_data = {
    "version": {"major": 1, "minor": 0},
    "type": "Multiple points",
    "points": output_points
}

with open('$SCRIPTPATH/test/input/$new_click_name', 'w') as f:
    json.dump(output_data, f, indent=2)

# 调试输出
left_count = len([p for p in output_points if p.get('name') == 'Left_IAC'])
right_count = len([p for p in output_points if p.get('name') == 'Right_IAC'])
print(f"Step {step}: {left_count} Left + {right_count} Right = {len(output_points)} total points")
EOF

                echo "  Generated: $new_image_name + $new_click_name (step $i)"
            done
        else
            echo "  WARNING: Missing click file $click_file, skipping case $base_name"
        fi
    fi
done

# Run the Docker container
echo "Running algorithm Docker container..."
docker run --rm \
    --memory=8g \
    --gpus all \
    -v "$SCRIPTPATH/test/input":/input \
    -v "$SCRIPTPATH/test/output":/output \
    $DOCKER_TAG

# Copy results to shared location for evaluation
echo "Copying results to shared location: $ALGORITHM_OUTPUT"
cp -r "$SCRIPTPATH/test/output/"* "$ALGORITHM_OUTPUT/"

if [ $? -eq 0 ]; then
    echo "Algorithm test completed successfully!"
    echo "Results copied to: $ALGORITHM_OUTPUT"
    echo "Ready for evaluation step."

    # 显示生成的文件统计
    echo ""
    echo "=== Test Summary ==="
    echo "Input images: $(ls $SCRIPTPATH/test/input/images/cbct/*.nii.gz | wc -l)"
    echo "Input clicks: $(ls $SCRIPTPATH/test/input/*.json | wc -l)"
    echo "Output segmentations: $(ls $ALGORITHM_OUTPUT/images/iac-segmentation/*.nii.gz 2>/dev/null | wc -l || echo 0)"
else
    echo "Error: Algorithm test failed with exit code $?"
    exit 1
fi