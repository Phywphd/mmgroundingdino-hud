#!/usr/bin/env python3
"""
MM-Grounding-DINO 推理脚本
对单张或多张图片进行目标检测并可视化结果
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import mmcv
import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint

# 导入 mmdet 以注册所有模块
import mmdet  # noqa
from mmdet.registry import MODELS, VISUALIZERS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


def parse_args():
    parser = argparse.ArgumentParser(description='MM-Grounding-DINO Inference')
    parser.add_argument('--config', type=str,
                        default='configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_my_dataset.py',
                        help='Config file path')
    parser.add_argument('--checkpoint', type=str,
                        default='work_dirs/my_dataset_finetune/best_coco_bbox_mAP_epoch_10.pth',
                        help='Checkpoint file path')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or folder path')
    parser.add_argument('--output', type=str, default='demo_results',
                        help='Output folder for visualization')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold for visualization')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--text-prompt', type=str, default=None,
                        help='Custom text prompt (default: use class names from config)')
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载配置
    cfg = Config.fromfile(args.config)

    # 获取类别名
    classes = cfg.metainfo.get('classes', ())
    if args.text_prompt:
        text_prompt = args.text_prompt
    else:
        # 使用类别名构建 text prompt
        text_prompt = ' . '.join(classes) + ' .'

    print(f"Classes: {classes}")
    print(f"Text prompt: {text_prompt}")

    # 构建模型
    print(f"\nLoading model from {args.checkpoint}...")
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    # 设置 dataset_meta
    model.dataset_meta = {'classes': classes}

    # 初始化可视化器
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = {'classes': classes}

    # 构建测试 pipeline
    test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
    # 移除 LoadAnnotations
    test_pipeline_cfg = [p for p in test_pipeline_cfg if p['type'] != 'LoadAnnotations']
    test_pipeline = Compose(test_pipeline_cfg)

    # 获取图片列表
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = list(input_path.glob('*.png')) + \
                      list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.jpeg'))

    print(f"Found {len(image_files)} images")
    print(f"Score threshold: {args.score_thr}")
    print("-" * 50)

    # 推理
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")

        # 准备数据
        data_info = {
            'img_path': str(img_path),
            'img_id': 0,
            'text': text_prompt,
            'custom_entities': False,
        }
        data = test_pipeline(data_info)

        # 准备 batch
        data['inputs'] = data['inputs'].unsqueeze(0).to(args.device)
        data['data_samples'] = [data['data_samples']]
        data['data_samples'][0].set_field(text_prompt, 'text')
        data['data_samples'][0].set_field(False, 'custom_entities')

        # 推理
        with torch.no_grad():
            results = model.predict(data['inputs'], data['data_samples'])

        result = results[0]

        # 读取原图用于可视化
        img = mmcv.imread(str(img_path))
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        # 可视化
        visualizer.add_datasample(
            name=img_path.stem,
            image=img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=args.score_thr,
            show=False,
            out_file=os.path.join(args.output, f'det_{img_path.name}')
        )

        # 打印检测结果
        pred_instances = result.pred_instances
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()

        count = 0
        for score, label, bbox in zip(scores, labels, bboxes):
            if score >= args.score_thr:
                class_name = classes[label] if label < len(classes) else f'class_{label}'
                print(f"  {class_name}: {score:.3f} at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                count += 1
        print(f"  Total: {count} detections")

    print(f"\n{'='*50}")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
