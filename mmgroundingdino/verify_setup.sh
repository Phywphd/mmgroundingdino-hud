#!/bin/bash
# 验证配置文件和数据集

source $(conda info --base)/etc/profile.d/conda.sh
conda activate groundingdino_py312

cd /home/kinova/ssd1/ljx/mmgroundingdino

echo "====================================="
echo "验证配置和数据集"
echo "====================================="

# 检查数据集文件
echo -e "\n[1/4] 检查数据集文件..."
if [ -f "/home/kinova/ssd1/ljx/my_dataset/annotations/instances_train.json" ]; then
    echo "✓ 训练集标注文件存在"
else
    echo "✗ 训练集标注文件不存在！"
    exit 1
fi

if [ -f "/home/kinova/ssd1/ljx/my_dataset/annotations/instances_val.json" ]; then
    echo "✓ 验证集标注文件存在"
else
    echo "✗ 验证集标注文件不存在！"
    exit 1
fi

# 检查图片目录
echo -e "\n[2/4] 检查图片目录..."
IMG_COUNT=$(ls /home/kinova/ssd1/ljx/my_dataset/images/*.png 2>/dev/null | wc -l)
echo "✓ 图片数量: ${IMG_COUNT}"

# 检查配置文件
echo -e "\n[3/4] 检查配置文件..."
if [ -f "configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_my_dataset.py" ]; then
    echo "✓ 配置文件存在"
else
    echo "✗ 配置文件不存在！"
    exit 1
fi

# 尝试加载配置
echo -e "\n[4/4] 验证配置文件格式..."
python -c "
from mmengine.config import Config
cfg = Config.fromfile('configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_my_dataset.py')
print('✓ 配置文件格式正确')
print(f'  - 类别数量: {len(cfg.class_name)}')
print(f'  - 类别名称: {cfg.class_name}')
print(f'  - 训练轮数: {cfg.max_epoch}')
print(f'  - 学习率: {cfg.optim_wrapper.optimizer.lr}')
" 2>&1

if [ $? -eq 0 ]; then
    echo -e "\n====================================="
    echo "✓ 所有检查通过！可以开始训练"
    echo "====================================="
    echo -e "\n运行以下命令开始训练:"
    echo "  ./train_my_dataset.sh"
else
    echo -e "\n✗ 配置验证失败，请检查配置文件"
    exit 1
fi
