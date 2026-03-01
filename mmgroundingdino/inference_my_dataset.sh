#!/bin/bash
# 使用微调后的模型进行推理测试

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate groundingdino

# 切换到 mmgroundingdino 目录
cd /home/kinova/ssd1/ljx/mmgroundingdino

# 配置文件
CONFIG="configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_my_dataset.py"

# 模型权重（训练完成后会自动保存为 best_bbox_mAP_50.pth 或 epoch_*.pth）
CHECKPOINT="work_dirs/my_dataset_finetune/best_bbox_mAP_50.pth"

# 测试图片
TEST_IMAGE="/home/kinova/ssd1/ljx/my_dataset/images/test_image.jpg"

# 类别名称（用空格或点分隔）
TEXTS="HUD_area. aircraft_cursor. x_left. y_up. y_down."

echo "====================================="
echo "使用微调模型进行推理"
echo "配置文件: ${CONFIG}"
echo "模型权重: ${CHECKPOINT}"
echo "测试图片: ${TEST_IMAGE}"
echo "====================================="

# 运行推理
python demo/image_demo.py ${TEST_IMAGE} \
    ${CONFIG} \
    --weights ${CHECKPOINT} \
    --texts "${TEXTS}" \
    --pred-score-thr 0.3

echo "====================================="
echo "推理完成！结果保存在 outputs/vis/ 目录"
echo "====================================="
