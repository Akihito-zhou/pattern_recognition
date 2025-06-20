# src/feature_extractor.py

import json
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys

# 再次添加路径，确保能找到config
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    PROCESSED_DATA_PATH,
    FEATURES_DIR,
)

# 尝试导入必要的库，如果失败则给出提示
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from ultralytics import YOLO
except ImportError:
    print("错误：必要的库未安装。")
    print("请确保你已经安装了 transformers, torch, ultralytics。")
    print("运行: pip install -r requirements.txt")
    sys.exit(1)


class FeatureExtractor:
    """
    一个用于从图像中提取文本特征的类。
    它会为每张图片生成两种特征：
    1. 基线特征：由BLIP生成的图像描述。
    2. 优化特征：BLIP描述 + YOLOv8检测到的物体列表。
    """
    def __init__(self, device: str = None):
        """
        初始化并加载所有需要的AI模型。
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps"
        else:
            self.device = device
        
        print(f"正在使用设备: {self.device}")
        print("正在加载模型，这可能需要一些时间，特别是第一次运行时需要下载模型权重...")
        
        # 加载BLIP模型用于图像描述
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        
        # 加载YOLOv8模型用于物体检测
        self.yolo_model = YOLO('yolov8n.pt') # yolov8n是最小最快的模型
        
        print("所有模型加载成功！")

    def extract_features_for_image(self, image_path: Path) -> tuple[str, str]:
        """
        为单个图像提取基线特征和优化特征。
        
        :param image_path: 图像文件的路径
        :return: 一个元组 (baseline_feature, optimized_feature)
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"警告：无法打开图像 {image_path}。跳过此图像。错误: {e}")
            return None, None
            
        # 1. 生成基线特征 (BLIP Caption)
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values
        generated_ids = self.blip_model.generate(pixel_values=pixel_values, max_length=50)
        baseline_feature = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # 2. 生成优化特征
        # 2a. 使用YOLO进行物体检测
        yolo_results = self.yolo_model(image, verbose=False) # verbose=False避免打印过多日志
        detected_objects = list(set([yolo_results[0].names[int(cls)] for cls in yolo_results[0].boxes.cls]))
        
        # 2b. 组合成优化特征字符串
        optimized_feature = (
            f"Scene Description: {baseline_feature}\n"
            f"Detected Objects: {detected_objects}"
        )
        
        return baseline_feature, optimized_feature

    def run(self):
        """
        主执行函数：读取数据，遍历图像，提取特征，并保存结果。
        """
        if not PROCESSED_DATA_PATH.exists():
            print(f"错误：处理后的数据文件不存在于 {PROCESSED_DATA_PATH}")
            print("请先运行 src/preprocess.py 脚本。")
            return
            
        print(f"正在从 {PROCESSED_DATA_PATH} 加载数据...")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        # 获取唯一的图像路径列表，避免重复处理同一张图片
        unique_image_paths = df[['image_id', 'image_path']].drop_duplicates().to_dict('records')
        
        baseline_features = {}
        optimized_features = {}
        
        print(f"开始为 {len(unique_image_paths)} 张唯一图像提取特征...")
        for item in tqdm(unique_image_paths, desc="Extracting Features"):
            image_id = item['image_id']
            image_path = Path(item['image_path'])
            
            base_feat, opt_feat = self.extract_features_for_image(image_path)
            
            if base_feat and opt_feat:
                baseline_features[image_id] = base_feat
                optimized_features[image_id] = opt_feat
        
        # 确保输出目录存在
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # 保存特征文件
        baseline_path = FEATURES_DIR / 'baseline_features.json'
        optimized_path = FEATURES_DIR / 'optimized_features.json'
        
        print(f"\n特征提取完成！正在保存结果...")
        with open(baseline_path, 'w') as f:
            json.dump(baseline_features, f, indent=4)
        print(f"基线特征已保存到: {baseline_path}")
        
        with open(optimized_path, 'w') as f:
            json.dump(optimized_features, f, indent=4)
        print(f"优化特征已保存到: {optimized_path}")
        
        print("\n所有特征已成功生成并保存！")


if __name__ == '__main__':
    extractor = FeatureExtractor()
    extractor.run()