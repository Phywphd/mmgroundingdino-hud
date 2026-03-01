#!/usr/bin/env python3
"""
将annotations.json按类别分割成多个单物体数据集
每个annotations{i}.json只包含对应category_id的标注
"""

import json
import os
from collections import defaultdict

def split_annotations_by_category(input_file, output_dir=None):
    """
    按类别分割COCO格式的annotations文件
    
    Args:
        input_file: 输入的annotations.json路径
        output_dir: 输出目录，默认与输入文件同目录
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # 读取原始annotations
    print(f"正在读取 {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取所有类别
    categories = data['categories']
    print(f"\n找到 {len(categories)} 个类别:")
    for cat in categories:
        print(f"  - ID {cat['id']}: {cat['name']}")
    
    # 按category_id分组annotations
    annotations_by_category = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_category[ann['category_id']].append(ann)
    
    # 统计每个类别使用的图像
    images_by_category = defaultdict(set)
    for ann in data['annotations']:
        images_by_category[ann['category_id']].add(ann['image_id'])
    
    # 为每个类别创建单独的annotations文件
    for category in categories:
        category_id = category['id']
        category_name = category['name']
        
        # 获取该类别的所有annotations
        category_annotations = annotations_by_category[category_id]
        
        # 获取该类别相关的图像ID
        relevant_image_ids = images_by_category[category_id]
        
        # 筛选出相关的图像
        relevant_images = [img for img in data['images'] if img['id'] in relevant_image_ids]
        
        # 创建新的annotations结构
        new_data = {
            'info': {
                'description': f"Dataset for {category_name} only",
                'version': data['info']['version'],
                'year': data['info']['year'],
                'contributor': data['info']['contributor'],
                'date_created': data['info']['date_created']
            },
            'licenses': data['licenses'],
            'images': relevant_images,
            'annotations': category_annotations,
            'categories': [category]  # 只包含当前类别
        }
        
        # 保存到新文件
        output_file = os.path.join(output_dir, f'annotations{category_id}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 创建 annotations{category_id}.json ({category_name})")
        print(f"  - 标注数量: {len(category_annotations)}")
        print(f"  - 图像数量: {len(relevant_images)}")
    
    # 创建统计摘要
    print("\n" + "="*60)
    print("分割完成统计:")
    print("="*60)
    print(f"原始数据集:")
    print(f"  - 总图像数: {len(data['images'])}")
    print(f"  - 总标注数: {len(data['annotations'])}")
    print(f"\n单物体数据集:")
    for category in categories:
        category_id = category['id']
        category_name = category['name']
        ann_count = len(annotations_by_category[category_id])
        img_count = len(images_by_category[category_id])
        print(f"  - annotations{category_id}.json ({category_name}): {ann_count} 标注, {img_count} 图像")

if __name__ == '__main__':
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'annotations.json')
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        exit(1)
    
    # 执行分割
    split_annotations_by_category(input_file, script_dir)
    
    print("\n✓ 所有文件已创建在:", script_dir)
