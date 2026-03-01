#!/bin/bash
# 微调 Grounding DINO 模型脚本 - 使用 my_dataset 数据集

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate groundingdino

# 切换到 mmgroundingdino 目录
cd /home/kinova/ssd1/ljx/mmgroundingdino

# 设置 GPU 数量（根据你的硬件调整）
GPUS=2

# 配置文件路径
CONFIG="configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_my_dataset.py"

# 工作目录（保存训练结果）
WORK_DIR="work_dirs/my_dataset_finetune"

# 创建工作目录
mkdir -p ${WORK_DIR}

echo "====================================="
echo "开始微调 Grounding DINO 模型"
echo "数据集: my_dataset (5个类别)"
echo "配置文件: ${CONFIG}"
echo "工作目录: ${WORK_DIR}"
echo "GPU 数量: ${GPUS}"
echo "====================================="

# 单卡训练
if [ ${GPUS} -eq 1 ]; then
    echo "使用单卡训练..."
    python tools/train.py ${CONFIG} --work-dir ${WORK_DIR}
else
    # 多卡训练
    echo "使用 ${GPUS} 卡训练..."
    ./tools/dist_train.sh ${CONFIG} ${GPUS} --work-dir ${WORK_DIR}
fi

echo "====================================="
echo "训练完成！"
echo "模型保存在: ${WORK_DIR}"
echo "====================================="
