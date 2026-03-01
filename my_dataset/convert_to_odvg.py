#!/usr/bin/env python3
"""
将 COCO 格式的标注文件转换为 Open-GroundingDino 的 ODVG 格式
同时生成 label_map.json 和 train/val 分割

输入: annotations.json (COCO 格式)
输出: 
    - train_odvg.jsonl (ODVG 格式训练集)
    - val_odvg.jsonl (ODVG 格式验证集)
    - val_coco.json (COCO 格式验证集，用于评估)
    - label_map.json (类别映射)
"""

import json
import jsonlines
import os
import random
from tqdm import tqdm
from collections import defaultdict

def coco_to_xyxy(bbox):
    """将 COCO 格式的 bbox [x, y, width, height] 转换为 [x1, y1, x2, y2]"""
    x, y, width, height = bbox
    x1 = round(x, 2)
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]

def load_coco_annotations(anno_path):
    """加载 COCO 格式标注文件"""
    with open(anno_path, 'r') as f:
        data = json.load(f)
    return data

def create_label_map(categories):
    """
    创建 label_map，将类别 ID 映射为从 0 开始的连续整数
    """
    # 按原始 ID 排序
    sorted_cats = sorted(categories, key=lambda x: x['id'])
    label_map = {}
    category_id_to_new_label = {}
    
    for new_label, cat in enumerate(sorted_cats):
        label_map[str(new_label)] = cat['name']
        category_id_to_new_label[cat['id']] = new_label
    
    return label_map, category_id_to_new_label

def convert_to_odvg(coco_data, category_id_to_new_label, category_id_to_name):
    """
    将 COCO 数据转换为 ODVG 格式
    """
    # 构建 image_id -> annotations 的映射
    img_id_to_annos = defaultdict(list)
    for anno in coco_data['annotations']:
        img_id_to_annos[anno['image_id']].append(anno)
    
    # 构建 image_id -> image_info 的映射
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    odvg_data = []
    for img_id, img_info in tqdm(img_id_to_info.items(), desc="Converting to ODVG"):
        annos = img_id_to_annos[img_id]
        
        instance_list = []
        for anno in annos:
            bbox = anno['bbox']
            bbox_xyxy = coco_to_xyxy(bbox)
            original_cat_id = anno['category_id']
            new_label = category_id_to_new_label[original_cat_id]
            category_name = category_id_to_name[original_cat_id]
            
            instance_list.append({
                "bbox": bbox_xyxy,
                "label": new_label,
                "category": category_name
            })
        
        odvg_data.append({
            "filename": img_info["file_name"],
            "height": img_info["height"],
            "width": img_info["width"],
            "detection": {
                "instances": instance_list
            }
        })
    
    return odvg_data

def create_coco_val_format(coco_data, val_img_ids, category_id_to_new_label):
    """
    创建用于评估的 COCO 格式验证集
    类别 ID 从 0 开始（与 ODVG 训练一致）
    """
    # 筛选验证集图片
    val_images = [img for img in coco_data['images'] if img['id'] in val_img_ids]
    
    # 筛选验证集标注
    val_annotations = []
    anno_id = 1
    for anno in coco_data['annotations']:
        if anno['image_id'] in val_img_ids:
            new_anno = anno.copy()
            new_anno['id'] = anno_id
            # 更新为新的类别 ID（从 0 开始）
            new_anno['category_id'] = category_id_to_new_label[anno['category_id']]
            val_annotations.append(new_anno)
            anno_id += 1
    
    # 创建新的类别列表（ID 从 0 开始）
    new_categories = []
    for cat in coco_data['categories']:
        new_cat = cat.copy()
        new_cat['id'] = category_id_to_new_label[cat['id']]
        new_categories.append(new_cat)
    
    val_coco = {
        "info": coco_data.get('info', {}),
        "licenses": coco_data.get('licenses', []),
        "images": val_images,
        "annotations": val_annotations,
        "categories": sorted(new_categories, key=lambda x: x['id'])
    }
    
    return val_coco

def split_dataset(odvg_data, val_ratio=0.2, seed=42):
    """
    将数据集分割为训练集和验证集
    """
    random.seed(seed)
    data_copy = odvg_data.copy()
    random.shuffle(data_copy)
    
    val_size = int(len(data_copy) * val_ratio)
    val_data = data_copy[:val_size]
    train_data = data_copy[val_size:]
    
    return train_data, val_data

def main():
    # 路径配置
    base_dir = "/home/kinova/ssd1/ljx/my_dataset"
    anno_path = os.path.join(base_dir, "annotations.json")
    
    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("COCO to ODVG 格式转换工具")
    print("=" * 60)
    
    # 1. 加载 COCO 标注
    print("\n1. 加载 COCO 标注文件...")
    coco_data = load_coco_annotations(anno_path)
    print(f"   - 图片数量: {len(coco_data['images'])}")
    print(f"   - 标注数量: {len(coco_data['annotations'])}")
    print(f"   - 类别数量: {len(coco_data['categories'])}")
    
    # 显示类别信息
    print("\n   类别列表:")
    for cat in coco_data['categories']:
        print(f"     - ID {cat['id']}: {cat['name']}")
    
    # 2. 创建 label_map
    print("\n2. 创建 label_map...")
    label_map, category_id_to_new_label = create_label_map(coco_data['categories'])
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"   label_map 已保存到: {label_map_path}")
    print(f"   映射关系: {label_map}")
    
    # 3. 转换为 ODVG 格式
    print("\n3. 转换为 ODVG 格式...")
    odvg_data = convert_to_odvg(coco_data, category_id_to_new_label, category_id_to_name)
    print(f"   转换完成，共 {len(odvg_data)} 条记录")
    
    # 4. 分割数据集
    print("\n4. 分割数据集 (80% 训练, 20% 验证)...")
    train_data, val_data = split_dataset(odvg_data, val_ratio=0.2, seed=42)
    print(f"   训练集: {len(train_data)} 张图片")
    print(f"   验证集: {len(val_data)} 张图片")
    
    # 5. 保存 ODVG 格式训练集
    train_odvg_path = os.path.join(output_dir, "train_odvg.jsonl")
    with jsonlines.open(train_odvg_path, mode='w') as writer:
        writer.write_all(train_data)
    print(f"\n5. 训练集 ODVG 已保存到: {train_odvg_path}")
    
    # 6. 保存 ODVG 格式验证集
    val_odvg_path = os.path.join(output_dir, "val_odvg.jsonl")
    with jsonlines.open(val_odvg_path, mode='w') as writer:
        writer.write_all(val_data)
    print(f"   验证集 ODVG 已保存到: {val_odvg_path}")
    
    # 7. 创建 COCO 格式验证集（用于评估）
    print("\n6. 创建 COCO 格式验证集（用于评估）...")
    val_filenames = {item['filename'] for item in val_data}
    val_img_ids = {img['id'] for img in coco_data['images'] if img['file_name'] in val_filenames}
    
    val_coco = create_coco_val_format(coco_data, val_img_ids, category_id_to_new_label)
    val_coco_path = os.path.join(output_dir, "val_coco.json")
    with open(val_coco_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    print(f"   验证集 COCO 格式已保存到: {val_coco_path}")
    
    # 8. 创建 Open-GroundingDino 数据集配置文件
    print("\n7. 创建数据集配置文件...")
    dataset_config = {
        "train": [
            {
                "root": os.path.join(base_dir, "images") + "/",
                "anno": train_odvg_path,
                "label_map": label_map_path,
                "dataset_mode": "odvg"
            }
        ],
        "val": [
            {
                "root": os.path.join(base_dir, "images") + "/",
                "anno": val_coco_path,
                "label_map": None,
                "dataset_mode": "coco"
            }
        ]
    }
    
    dataset_config_path = os.path.join(output_dir, "datasets_config.json")
    with open(dataset_config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)
    print(f"   数据集配置已保存到: {dataset_config_path}")
    
    # 9. 打印使用说明
    print("\n" + "=" * 60)
    print("转换完成！生成的文件：")
    print("=" * 60)
    print(f"  - label_map.json      : 类别映射文件")
    print(f"  - train_odvg.jsonl    : ODVG 格式训练集 ({len(train_data)} 张)")
    print(f"  - val_odvg.jsonl      : ODVG 格式验证集 ({len(val_data)} 张)")
    print(f"  - val_coco.json       : COCO 格式验证集（用于评估）")
    print(f"  - datasets_config.json: Open-GroundingDino 数据集配置")
    
    print("\n" + "=" * 60)
    print("下一步：使用 Open-GroundingDino 进行训练")
    print("=" * 60)
    print(f"""
1. 进入 Open-GroundingDino 目录:
   cd /home/kinova/ssd1/ljx/open_grounding_dino

2. 下载预训练模型（如果还没有）:
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

3. 下载 BERT 模型:
   确保有 bert-base-uncased 模型

4. 开始训练:
   bash train_dist.sh 1 config/cfg_odvg.py {dataset_config_path} ./output
""")

if __name__ == "__main__":
    main()
