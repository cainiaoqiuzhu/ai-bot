# encoding:utf-8
from typing import Union, List

import requests
import json
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from file_processing import load_embedding_mode
from DashScopeClient import DashScopeClient
from langchain_core.tools import tool
from config import qwen_setting, yuanfenju
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool

YUANFENJU_API_KEY = yuanfenju['yuanfenju_api']

qwenOpenAIModel = DashScopeClient(qwen_setting['api_key'], qwen_setting['qwen_api'])
embeddingModel = load_embedding_mode("bge_small")


@register_tool("search")
class Search(BaseToolWithFileAccess):
    description = '只有需要了解实时信息或者不知道才会使用这个工具'
    parameters = [
        {
            'name':'query',
            'type':'string',
            'description':'用户提问',
            'required':True
        }
    ]

    def call(self, params: Union[str, dict], files: List[str] = None, **kwargs) -> str:
        super().call(params=params, files=files)
        params = self._verify_json_format_args(params)

        query = params['query']
        serp = SerpAPIWrapper()
        try:
            result = serp.run(query)
            print("实时搜索结果:", result)
            return result
        except Exception as e:
            return f"调用搜索工具时出错了: {str(e)}"


def document_to_dict(document):
    return {
        'metadata': document.metadata,
        'page_content': document.page_content
    }

@register_tool("get_info_from_local_db")
class getInfoFromLocalDb(BaseToolWithFileAccess):
    description = '只有回答与2025年运势或者蛇年运势相关的问题的时候才会使用这个工具。'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': '用户提问',
            'required': True
        }
    ]

    def call(self, params: Union[str, dict], files: List[str] = None, **kwargs) -> str:
        super().call(params=params, files=files)
        params = self._verify_json_format_args(params)
        query = params['query']
        client = Qdrant(
            QdrantClient(path="./db/local_qdrant"),
            "yunshi_2025",
            embeddings=embeddingModel,
        )
        retriever = client.as_retriever(search_type="mmr")
        result = retriever.get_relevant_documents(query)
        serializable_result = [document_to_dict(doc) for doc in result]
        return json.dumps(serializable_result, ensure_ascii=False, indent=4)

@register_tool("bazi_compute")
class BaziCompute(BaseToolWithFileAccess):
    description = '只有做八字排盘的时候才会使用这个工具，需要输入用户姓名和出生年月时，如果缺少用户姓名和出生年月则不可用'
    parameters = [
        {
            'name': 'name',
            'type': 'string',
            'description': '姓名',
            'required': True
        },
        {
            'name': 'sex',
            'type': 'int',
            'description': '性别，0为男，1为女',
            'required': True
        },
        {
            'name': 'type',
            'type': 'int',
            'description': '历类型，0为农历，1为公历',
            'required': True
        },
        {
            'name': 'year',
            'type': 'int',
            'description': '出生年',
            'required': True
        },
        {
            'name': 'month',
            'type': 'int',
            'description': '出生月',
            'required': True
        },
        {
            'name': 'day',
            'type': 'int',
            'description': '出生日',
            'required': True
        },
        {
            'name': 'hours',
            'type': 'int',
            'description': '出生时',
            'required': True
        },
        {
            'name': 'minute',
            'type': 'int',
            'description': '出生分',
            'required': True
        },
        {
            'name': 'sect',
            'type': 'int',
            'description': '流派，1：晚子时日柱算明天，2：晚子时日柱算当天，默认1',
            'required': False
        },
        {
            'name': 'zhen',
            'type': 'int',
            'description': '是否真太阳时，1：考虑真太阳时，2：不考虑，默认2',
            'required': False
        },
        {
            'name': 'province',
            'type': 'string',
            'description': '省份，考虑真太阳时时必填',
            'required': False
        },
        {
            'name': 'city',
            'type': 'string',
            'description': '城市，考虑真太阳时时必填',
            'required': False
        },
        {
            'name': 'lang',
            'type': 'string',
            'description': '语言，默认为zh-cn',
            'required': False
        },
        {
            'name': 'factor',
            'type': 'int',
            'description': '调整因子，0：不作调整，1：需要作调整，默认0',
            'required': False
        }
    ]

    def call(self, params: Union[str, dict], files: List[str] = None, **kwargs) -> str:
        super().call(params=params, files=files)
        params = self._verify_json_format_args(params)
        url = f"https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"

        name = params['name']
        sex = params['sex']
        type_ = params['type']
        year = params['year']
        month = params['month']
        day = params['day']
        hours = params['hours']
        minute = params['minute']
        sect = params.get('sect', 1)
        zhen = params.get('zhen', 2)
        province = params.get('province', '')
        city = params.get('city', '')
        lang = params.get('lang', 'zh-cn')
        factor = params.get('factor', '0')

        # 构造请求参数字典
        query_params = {
            'api_key': YUANFENJU_API_KEY,
            'name': name,
            'sex': sex,
            'type': type_,
            'year': year,
            'month': month,
            'day': day,
            'hours': hours,
            'minute': minute,
            'sect': sect,
            'zhen': zhen,
            'province': province,
            'city': city,
            'lang': lang,
            'factor': factor
        }
        result = requests.get(url, params=query_params)
        if result.status_code == 200:
            json_result = result.json()
            print("八字查询结果:", json_result)
            return f"八字为: {json_result['data']['bazi_info']['bazi']}"
        else:
            return "技术错误，请告诉用户稍后再试。"

@register_tool("yao_yi_gua")
class yaoyigua(BaseToolWithFileAccess):
    description = '当用户提到占卜、抽签、摇一卦、或者求卦相关内容时，使用此工具来进行预测'

    def call(self, params: Union[str, dict], files: List[str] = None, **kwargs) -> str:
        url = f"https://api.yuanfenju.com/index.php/v1/Zhanbu/yaogua"
        query_params = {
            'api_key': YUANFENJU_API_KEY
        }
        result = requests.get(url, params=query_params)
        if result.status_code == 200:
            print("====返回数据====")
            print(result.json())
            returnstring = result.json()
            image = returnstring["data"]["image"]
            print("卦图片:", image)
            return returnstring
        else:
            return "技术错误，请告诉用户稍后再试。"

@register_tool("jie_meng")
class jiemeng(BaseToolWithFileAccess):
    description = '只有用户想要解梦的时候才会使用这个工具，需要输入用户梦境的内容，如果缺少用户的梦境内容则不可用'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': '用户提问',
            'required': True
        }
    ]

    def call(self, params: Union[str, dict], files: List[str] = None, **kwargs) -> str:
        super().call(params=params, files=files)
        params = self._verify_json_format_args(params)
        url = f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
        query = "根据内容提取一个关键词，只返回关键词，内容为:{topic}".format(topic=params['query'])
        keyword = qwenOpenAIModel(query)
        print("提取的关键词是:", keyword)
        result = requests.post(url, data={
            "api_key": YUANFENJU_API_KEY,
            "title_zhougong": keyword
        })
        if result.status_code == 200:
            print("====返回数据====")
            print(result.json())
            returnstring = json.loads(result.text)
            return returnstring
        else:
            return "技术错误，请告诉用户稍后再试。"

def jiemeng(query:str):
    """"""
    api_key = YUANFENJU_API_KEY
    url = f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    prompt = PromptTemplate.from_template("根据内容提取一个关键词，只返回关键词，内容为:{topic}")
    prompt_value = prompt.invoke({"topic":query})
    keyword = qwenOpenAIModel.invoke(prompt_value)
    print("提取的关键词是:",keyword)
    result = requests.post(url, data={
        "api_key": api_key,
        "title_zhougong": keyword
    })
    if result.status_code == 200:
        print("====返回数据====")
        print(result.json())
        returnstring = json.loads(result.text)
        return returnstring
    else:
        return "技术错误，请告诉用户稍后再试。"