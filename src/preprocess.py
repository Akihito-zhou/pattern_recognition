# src/preprocess.py

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# 将 src 目录添加到 Python 路径，以便可以导入 config
# 这样无论你在哪个目录下运行这个脚本，都能找到config模块
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    RAW_QUESTIONS_PATH,
    RAW_ANNOTATIONS_PATH,
    RAW_IMAGES_DIR,
    PROCESSED_DATA_PATH,
)

def create_yes_no_dataset():
    """
    加载原始VQA数据，筛选出答案为 "yes" 或 "no" 的问答对，
    并将其处理成一个结构化的DataFrame。
    """
    # 1. 检查原始文件是否存在
    if not all([RAW_QUESTIONS_PATH.exists(), RAW_ANNOTATIONS_PATH.exists(), RAW_IMAGES_DIR.exists()]):
        print("错误：一个或多个原始数据文件/文件夹不存在。请检查 config.py 中的路径是否正确，以及数据是否已下载。")
        print(f"检查路径: \n- {RAW_QUESTIONS_PATH}\n- {RAW_ANNOTATIONS_PATH}\n- {RAW_IMAGES_DIR}")
        return

    print("开始预处理数据...")
    
    # 2. 加载原始JSON文件
    print("加载原始问题和注释文件...")
    with open(RAW_QUESTIONS_PATH, 'r') as f:
        questions_data = json.load(f)
    with open(RAW_ANNOTATIONS_PATH, 'r') as f:
        annotations_data = json.load(f)

    # 3. 为了快速查找，将注释转换为以 question_id 为键的字典
    # 这是提高处理效率的关键步骤
    print("正在构建注释查找表...")
    annotations_map = {ann['question_id']: ann for ann in annotations_data['annotations']}
    
    # 4. 遍历问题，筛选并构建新数据集
    print("遍历问题并筛选Yes/No问答对...")
    processed_records = []
    for question in tqdm(questions_data['questions'], desc="Processing questions"):
        question_id = question['question_id']
        
        # 查找对应的注释
        if question_id in annotations_map:
            annotation = annotations_map[question_id]
            answer = annotation['multiple_choice_answer']
            
            # 筛选答案为 "yes" 或 "no" 的数据
            if answer in ['yes', 'no']:
                # 构建图像文件名 (MS COCO 格式)
                # 例如 image_id 123 -> COCO_val2014_000000000123.jpg
                image_id = question['image_id']
                image_filename = f"COCO_val2014_{image_id:012d}.jpg"
                image_path = RAW_IMAGES_DIR / image_filename
                
                # 检查图像文件是否真的存在
                if image_path.exists():
                    record = {
                        'question_id': question_id,
                        'image_id': image_id,
                        'image_path': str(image_path), # 存为字符串，方便CSV读写
                        'question': question['question'],
                        'answer': answer,
                        'answer_label': 1 if answer == 'yes' else 0 # 转换为机器学习友好的标签
                    }
                    processed_records.append(record)

    if not processed_records:
        print("错误：没有找到任何 'yes'/'no' 的问答对。请确认数据集文件是否正确。")
        return

    # 5. 转换为Pandas DataFrame并保存
    df = pd.DataFrame(processed_records)
    
    # 确保输出目录存在
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n处理完成！共找到 {len(df)} 条 Yes/No 问答对。")
    print(f"正在保存处理后的数据到: {PROCESSED_DATA_PATH}")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print("\n数据预处理成功！")
    print("生成的文件预览 (前5行):")
    print(df.head())
    print(f"\n各类别数量:\n{df['answer'].value_counts()}")


if __name__ == '__main__':
    create_yes_no_dataset()