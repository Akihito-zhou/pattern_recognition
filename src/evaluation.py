# src/evaluation.py

def parse_llm_answer(llm_response: str) -> str | None:
    """
    从LLM可能返回的复杂文本中，解析出干净的 "yes" 或 "no"。

    :param llm_response: LLM返回的原始字符串。
    :return: "yes", "no" 或 None (如果无法解析)。
    """
    if not isinstance(llm_response, str):
        return None
        
    response_lower = llm_response.lower().strip()
    
    # 最理想情况，答案就在结尾
    if response_lower.endswith("final answer: yes"):
        return "yes"
    if response_lower.endswith("final answer: no"):
        return "no"

    # 兼容简单回答
    if response_lower == "yes":
        return "yes"
    if response_lower == "no":
        return "no"
        
    # 尝试从文本中寻找关键词
    if "yes" in response_lower.split() and "no" not in response_lower.split():
        return "yes"
    if "no" in response_lower.split() and "yes" not in response_lower.split():
        return "no"
        
    # 如果都无法判断，则返回None
    return None

def calculate_accuracy(predictions: list, ground_truths: list) -> float:
    """
    计算预测的准确率。
    """
    correct = 0
    total = len(predictions)
    if total == 0:
        return 0.0
        
    for pred, true in zip(predictions, ground_truths):
        if pred == true:
            correct += 1
            
    return (correct / total) * 100