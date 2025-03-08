# encoding:utf-8
from langchain_core.messages import BaseMessageChunk
from langchain_core.outputs import ChatGeneration, Generation, ChatGenerationChunk
from openai import OpenAI
from langchain_core.language_models import LLM
from config import model_setting


client = OpenAI(
    api_key=model_setting['api_key'],
    base_url=model_setting['openai_api']
)

def call_qwen_api(question, max_tokens=512):
    chat_response = client.chat.completions.create(
        model="deepseek-R1-671B",
        messages=[
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=max_tokens,
        extra_body={
            "repetition_penalty": 1.05,
        },
        stream=True
    )
    result = ""
    for chunk in chat_response:
        # 从流中提取部分内容
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            content = getattr(delta, 'content', "")  # 使用属性访问方式
            if content:  # 确保content不是None或空字符串
                print(content, end="", flush=True)  # 实时打印
                result += content  # 拼接完整内容
    print()  # 输出换行
    return result  # 返回完整结果



def call_qwen_msg(messages, max_tokens=512):
    chat_response = client.chat.completions.create(
        model="Qwen2.5-72B-Instruct",
        messages=messages,
        temperature=0.7,
        top_p=0.8,
        max_tokens=max_tokens,
        extra_body={
            "repetition_penalty": 1.05,
        },
        # stream=True # 启动流式模式
    )
    # return chat_response
    return chat_response.choices[0].message.content

class QwenOpenAIModel(LLM):
    def __init__(self):
        super().__init__()  # 确保初始化父类

    def _call(self, prompt: str, **kwargs) -> str:
        #使用 call_qwen_api 函数获取模型的响应
        return call_qwen_msg(
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

    @property
    def _llm_type(self):
        # 需要返回模型类型的字符串                                                                                            
        return "QwenOpenAI"  # 您可以根据需要更改为其他类型

if __name__ == '__main__':
    print(call_qwen_api(question="16是什么数"))

