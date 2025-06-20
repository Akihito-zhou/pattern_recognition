# src/llm_handler.py

import sys
from pathlib import Path
import time

# 确保能找到config
sys.path.append(str(Path(__file__).parent.parent))
# --- 导入配置变量 ---
from src.config import LLM_API_KEY, LLM_MODEL_NAME 

# --- 导入新的库 ---
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError:
    print("错误: google-generativeai 库未安装。请运行 pip install -r requirements.txt")
    sys.exit(1)


class LLMHandler:
    """
    处理所有与Google Gemini API交互的类。
    """
    # 重试逻辑的常量
    MAX_RETRIES = 5
    INITIAL_RETRY_DELAY_SECONDS = 1  # 首次重试的等待时间
    MAX_RETRY_DELAY_SECONDS = 60     # 重试等待时间上限
    # 付费版用户可以设置一个非常小的默认请求间延时，甚至为0
    # 如果API响应很快且您的配额足够高，可以设为0以最大化吞吐量
    # 设为0.05到0.1秒可以作为一个温和的节流阀，避免过于激进地冲击API端点
    DEFAULT_INTER_REQUEST_DELAY_SECONDS = 0.05

    def __init__(self):
        # --- 初始化 ---
        if not LLM_API_KEY or LLM_API_KEY == "": # 修正占位符检查
            raise ValueError("Google API 密钥未在 config.py 中设置。")
        
        # 配置API密钥
        genai.configure(api_key=LLM_API_KEY)
        
        # 设置生成参数
        self.generation_config = {
            "temperature": 0, # 设置为0以获得更具确定性的输出
            "max_output_tokens": 150,
        }
        
        # 设置安全设置，避免因内容审查而拒绝回答
        # 这是使用Gemini时的一个好习惯
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # 初始化模型
        self.model = genai.GenerativeModel(
            model_name=LLM_MODEL_NAME,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        print(f"Gemini处理器初始化成功，使用模型: {LLM_MODEL_NAME}")

    def build_prompt(self, visual_feature: str, question: str, prompt_type: str) -> str:
        # --- 根据指定的类型构建完整的Prompt ---
        if prompt_type == 'baseline':
            return (
                f"Based on the following image information, answer the question with only \"Yes\" or \"No\".\n\n"
                f"Image Information: \"{visual_feature}\"\n"
                f"Question: \"{question}\"\n\n"
                f"Answer:"
            )
        elif prompt_type == 'optimized_cot':
            return (
                f"You are a logical reasoning expert. Analyze the provided image information to answer the question. "
                f"First, provide a brief reasoning, then conclude with \"Final Answer: Yes\" or \"Final Answer: No\".\n\n"
                f"---\n"
                f"Example 1:\n"
                f"Image Information: \"Scene Description: A dog is sleeping on a red sofa.\"\n"
                f"Question: \"Is the sofa blue?\"\n"
                f"Reasoning: The description explicitly states the sofa is red. Therefore, it is not blue.\n"
                f"Final Answer: No\n\n"
                f"---\n"
                f"Your Task:\n"
                f"Image Information: \"{visual_feature}\"\n"
                f"Question: \"{question}\"\n"
                f"Reasoning:"
            )
        else:
            raise ValueError(f"未知的 prompt 类型: {prompt_type}")

    def query(self, prompt: str) -> str | None:
        current_retry_delay = self.INITIAL_RETRY_DELAY_SECONDS
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt == 0 and self.DEFAULT_INTER_REQUEST_DELAY_SECONDS > 0:
                    time.sleep(self.DEFAULT_INTER_REQUEST_DELAY_SECONDS)
                
                response = self.model.generate_content(prompt)

                # 1. 检查是否因安全原因被阻止
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason_name = "Unknown"
                    if hasattr(response.prompt_feedback.block_reason, 'name'):
                         reason_name = response.prompt_feedback.block_reason.name
                    print(f"警告：模型响应因安全原因被阻止。原因: {reason_name}. Prompt: \"{prompt[:100]}...\"")
                    return f"Blocked by model: {reason_name}" # 不重试安全阻止

                # 2. 尝试获取文本内容
                # response.text 会在内容被阻止时抛出 ValueError
                # 如果 response.parts 为空且未被阻止，response.text 可能是空字符串
                if response.parts:
                    return response.text
                else:
                    # 到这里意味着 response.parts 为空，且 prompt_feedback 没有报告阻止
                    # 这可能是模型生成了一个合法的空回复，或者是一个未明确报告的阻止
                    print(f"警告：模型返回了内容，但内容部分为空，且未明确说明阻止原因。Prompt: \"{prompt[:100]}...\"")
                    return "Empty content from model" # 视为特定类型的失败，不重试

            except ValueError as ve: # 通常由 response.text 在内容被阻止时抛出
                print(f"错误：从模型响应中提取文本时发生ValueError: {ve}. Prompt: \"{prompt[:100]}...\"")
                # 再次检查 prompt_feedback，以防万一
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     reason_name = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else "Unknown"
                     print(f"补充：ValueError可能与安全阻止有关。原因: {reason_name}")
                     return f"Blocked by model (detected via ValueError): {reason_name}" # 不重试
                return "Error extracting text from model response" # 其他ValueError，不重试

            except (google_exceptions.ResourceExhausted,
                    google_exceptions.DeadlineExceeded,
                    google_exceptions.ServiceUnavailable,
                    google_exceptions.InternalServerError) as e:
                
                error_type = type(e).__name__
                if attempt == self.MAX_RETRIES:
                    print(f"错误：API调用失败 ({error_type})，已达最大重试次数 ({self.MAX_RETRIES})。错误: {e}")
                    return None
                
                print(f"警告：API调用暂时失败 ({error_type})。将在 {current_retry_delay:.2f} 秒后重试 (尝试 {attempt + 1}/{self.MAX_RETRIES + 1})...")
                time.sleep(current_retry_delay)
                current_retry_delay = min(current_retry_delay * 2, self.MAX_RETRY_DELAY_SECONDS) # 指数退避
            
            except google_exceptions.GoogleAPICallError as e: # 其他 Google API 调用错误
                # 这类错误通常指示更根本的问题（如认证、无效参数），一般不可重试
                print(f"错误：发生Google API调用错误: {e}. Prompt: \"{prompt[:100]}...\"")
                return None 

            except Exception as e: # 捕获其他所有意外错误
                print(f"错误：与Gemini API交互时发生未知错误: {e}. Prompt: \"{prompt[:100]}...\"")
                if attempt == self.MAX_RETRIES:
                    print(f"未知错误已达最大重试次数。")
                    return None
                # 对于未知错误，谨慎重试一次或两次可能有助于处理瞬时问题
                print(f"未知错误。将在 {current_retry_delay:.2f} 秒后重试 (尝试 {attempt + 1}/{self.MAX_RETRIES + 1})...")
                time.sleep(current_retry_delay)
                current_retry_delay = min(current_retry_delay * 2, self.MAX_RETRY_DELAY_SECONDS)

        # 如果所有重试都失败了
        print(f"错误：所有 {self.MAX_RETRIES + 1} 次尝试均失败。")
        return None