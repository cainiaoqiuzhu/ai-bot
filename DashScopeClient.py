# encoding:utf-8
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
import json
from config import qwen_setting

class DashScopeClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def chat(self, messages: list, model: str = "qwen-plus"):
        """
        使用聊天模型发送消息并获取返回内容
        :param message:
        :param model:
        :return:
        """
        try:
            completion = self.client.chat.completions.create(
                model = model,
                messages = messages
            )

            return completion.model_dump_json()
        except Exception as e:
            raise Exception(f"请求失败: {str(e)}")

    def extract_content_from_response(self, response: str):
        try:
            response_dict = json.loads(response)
            content = response_dict['choices'][0]['message']['content']
            return content
        except Exception as e:
            raise Exception(f"响应数据格式错误，缺少字段：{e}")

    def __call__(self, query: str):
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ]

        response = self.chat(messages)
        return self.extract_content_from_response(response)


def get_model_response(query: str):
    client = DashScopeClient(qwen_setting["api_key"], qwen_setting["qwen_api"])

    # 定义聊天信息
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': query}
    ]
    try:
        response = client.chat(messages)
        content = client.extract_content_from_response(response)
        return content
    except Exception as e:
        raise Exception(f"请求失败: {str(e)}")


if __name__ == '__main__':
    query = "你是谁"
    # result = get_model_response(query)
    # print(result)
    chatmodel = DashScopeClient(qwen_setting['api_key'], qwen_setting['qwen_api'])
    result = chatmodel(query)
    print(result)