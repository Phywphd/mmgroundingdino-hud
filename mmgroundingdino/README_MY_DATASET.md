# 使用 my_dataset 微调 Grounding DINO 模型

## 📁 准备工作（已完成）

✅ 数据集路径: `/home/kinova/ssd1/ljx/my_dataset/`
✅ 图片数量: 775 张
✅ 类别数量: 5 个
✅ 类别名称: HUD_area, aircraft_cursor, x_left, y_up, y_down
✅ Conda 环境: groundingdino_py312

## 🚀 快速开始

### 1. 开始训练

```bash
cd /home/kinova/ssd1/ljx/mmgroundingdino
./train_my_dataset.sh
```

**训练配置:**
- 训练轮数: 50 epochs
- 学习率: 0.0001
- 验证间隔: 每 2 个 epoch
- 保存间隔: 每 5 个 epoch
- 策略: 冻结 backbone 和 language_model，仅训练检测头

### 2. 监控训练

训练日志和权重保存在:
```
/home/kinova/ssd1/ljx/mmgroundingdino/work_dirs/my_dataset_finetune/
```

查看训练日志:
```bash
tail -f work_dirs/my_dataset_finetune/*.log
```

### 3. 评估模型

训练完成后评估模型性能:
```bash
./test_my_dataset.sh
```

### 4. 推理测试

使用训练好的模型进行推理:
```bash
# 修改 inference_my_dataset.sh 中的 TEST_IMAGE 路径
./inference_my_dataset.sh
```

或者直接使用命令:
```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate groundingdino_py312
cd /home/kinova/ssd1/ljx/mmgroundingdino

python demo/image_demo.py /path/to/your/image.jpg \
    configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_my_dataset.py \
    --weights work_dirs/my_dataset_finetune/best_bbox_mAP_50.pth \
    --texts "HUD_area. aircraft_cursor. x_left. y_up. y_down." \
    --pred-score-thr 0.3
```

## 📊 训练参数调整

如需调整训练参数，编辑配置文件:
```bash
vim configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_my_dataset.py
```

常用调整:
- `max_epoch`: 训练轮数（当前 50）
- `lr`: 学习率（当前 0.0001）
- `train_cfg.val_interval`: 验证间隔（当前 2）
- `lr_mult`: backbone/language_model 学习率倍率（当前 0.0 表示冻结）

## 🔧 多GPU训练

如果有多个GPU，编辑 `train_my_dataset.sh`:
```bash
GPUS=2  # 改为你的GPU数量
```

## 📝 输出文件

训练完成后会生成:
- `best_bbox_mAP_50.pth` - 验证集上mAP最高的模型
- `epoch_*.pth` - 每5个epoch保存的检查点
- `*.log.json` - 训练日志（JSON格式）
- `*.log` - 训练日志（文本格式）

## ⚠️ 注意事项

1. 确保有足够的GPU显存（建议至少8GB）
2. 首次训练会自动下载预训练权重（约700MB）
3. 训练时间取决于硬件，单卡约需要几小时
4. 可以使用 `Ctrl+C` 安全停止训练，下次可以恢复

## 🐛 故障排除

**显存不足:** 减小 batch size，编辑配置文件中的 `train_dataloader.batch_size`

**依赖缺失:** 安装多模态依赖
```bash
conda activate groundingdino_py312
cd /home/kinova/ssd1/ljx/mmgroundingdino
pip install -r requirements/multimodal.txt
```

**权重下载失败:** 手动下载预训练权重并修改配置文件中的 `load_from` 路径
