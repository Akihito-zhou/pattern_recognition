# src/config.py
from dotenv import load_dotenv
import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 原始数据路径
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
RAW_QUESTIONS_PATH = RAW_DATA_DIR / 'v2_OpenEnded_mscoco_val2014_questions.json'
RAW_ANNOTATIONS_PATH = RAW_DATA_DIR / 'v2_mscoco_val2014_annotations.json'
RAW_IMAGES_DIR = RAW_DATA_DIR / 'val2014'

# 处理后数据的路径
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / 'yes_no_vqa_data.csv'

# 结果保存路径
RESULTS_DIR = DATA_DIR / 'results'

# 特征文件路径
FEATURES_DIR = DATA_DIR / 'features'

# AI 模型配置
LLM_API_KEY = os.getenv('LLM_API_KEY')  # 从环境变量获取API密钥
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')