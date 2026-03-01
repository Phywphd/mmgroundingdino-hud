#!/bin/bash
# 评估微调后的模型在验证集上的性能

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate groundingdino

# 切换到 mmgroundingdino 目录
cd /home/kinova/ssd1/ljx/mmgroundingdino

# 配置文件
CONFIG="configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_my_dataset.py"

# 模型权重
CHECKPOINT="work_dirs/my_dataset_finetune/best_bbox_mAP_50.pth"

echo "====================================="
echo "评估微调模型性能"
echo "配置文件: ${CONFIG}"
echo "模型权重: ${CHECKPOINT}"
echo "====================================="

# 运行评估
python tools/test.py ${CONFIG} ${CHECKPOINT}

echo "====================================="
echo "评估完成！"
echo "====================================="
