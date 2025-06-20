# run_experiments.py

import pandas as pd
import json
from tqdm import tqdm
import sys
from pathlib import Path

# 确保能找到src目录下的模块
sys.path.append(str(Path(__file__).parent))
from src.config import PROCESSED_DATA_PATH, FEATURES_DIR, RESULTS_DIR
from src.llm_handler import LLMHandler
from src.evaluation import parse_llm_answer, calculate_accuracy

def main():
    """
    主函数，用于执行所有VQA实验。
    """
    # 1. 加载所有需要的数据
    print("正在加载数据和特征...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    with open(FEATURES_DIR / 'baseline_features.json', 'r') as f:
        baseline_features = json.load(f)
    with open(FEATURES_DIR / 'optimized_features.json', 'r') as f:
        optimized_features = json.load(f)
        
    # 将image_id转换为字符串以匹配JSON的键
    df['image_id'] = df['image_id'].astype(str)
    
    # 创建一个测试子集以快速调试 (建议初次运行时使用)
    # test_df = df.head(10)

    # 正式运行时，我们从全部数据中随机抽取一个1000条的子集来进行实验
    # random_state=42 确保每次抽取的样本都是一样的，保证了实验的可复现性
    print(f"从 {len(df)} 条总数据中，随机抽取1000条作为本次实验的测试集。")
    test_df = df.sample(n=1000, random_state=42)
    
    
    # 2. 初始化LLM处理器
    print("正在初始化LLM处理器...")
    try:
        llm_handler = LLMHandler()
    except ValueError as e:
        print(f"错误: {e}")
        return

    # 3. 定义实验配置
    experiments = [
        {'name': '1_baseline', 'feature_type': 'baseline', 'prompt_type': 'baseline'},
        {'name': '2_feature_optimized', 'feature_type': 'optimized', 'prompt_type': 'baseline'},
        {'name': '3_prompt_optimized', 'feature_type': 'baseline', 'prompt_type': 'optimized_cot'},
        {'name': '4_combined_optimized', 'feature_type': 'optimized', 'prompt_type': 'optimized_cot'},
    ]
    
    all_results = []
    summary = []

    # 4. 循环执行所有实验
    for exp in experiments:
        print(f"\n--- 正在运行实验: {exp['name']} ---")
        
        predictions = []
        ground_truths = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Experiment {exp['name']}"):
            image_id = row['image_id']
            question = row['question']
            true_answer = row['answer']
            
            # 选择特征
            if exp['feature_type'] == 'baseline':
                visual_feature = baseline_features.get(image_id, "")
            else:
                visual_feature = optimized_features.get(image_id, "")
            
            if not visual_feature:
                continue

            # 构建并查询prompt
            prompt = llm_handler.build_prompt(visual_feature, question, exp['prompt_type'])
            llm_response = llm_handler.query(prompt)
            
            # 解析并记录结果
            predicted_answer = parse_llm_answer(llm_response)
            
            predictions.append(predicted_answer)
            ground_truths.append(true_answer)
            
            all_results.append({
                'experiment': exp['name'],
                'question_id': row['question_id'],
                'question': question,
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'llm_raw_response': llm_response
            })
        
        # 5. 计算并打印当前实验的准确率
        accuracy = calculate_accuracy([p for p in predictions if p], [gt for p, gt in zip(predictions, ground_truths) if p])
        summary.append({'experiment': exp['name'], 'accuracy': accuracy})
        print(f"实验 '{exp['name']}' 完成。准确率: {accuracy:.2f}%")

    # 6. 保存所有结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logs_path = RESULTS_DIR / 'experiment_logs.json'
    summary_path = RESULTS_DIR / 'summary.csv'
    
    with open(logs_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_path, index=False)
    
    print("\n--- 所有实验已完成！---")
    print(f"详细日志已保存到: {logs_path}")
    print(f"结果摘要已保存到: {summary_path}")
    print("\n最终结果:")
    print(summary_df)

if __name__ == '__main__':
    main()