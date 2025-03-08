# encoding:utf-8
import asyncio
import re
import shutil
from typing import List
import uuid
import os

import edge_tts
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.websockets import WebSocket, WebSocketDisconnect
from qwen_agent.agents import FnCallAgent
import redis

from file_processing import load_embedding_mode
from my_tools import Search, getInfoFromLocalDb, BaziCompute, yaoyigua, jiemeng
from DashScopeClient import DashScopeClient
from config import qwen_setting

qwenOpenAIModel = DashScopeClient(qwen_setting['api_key'], qwen_setting['qwen_api'])
myQwenModel = {
        'model_type': 'qwenvl_oai',
        'model': 'qwen-plus',
        'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': qwen_setting['api_key'],
        'generate_cfg': dict(top_p=0.8, max_input_tokens=15000, temperature=0.7, max_retries=10)
}

def filter_text(text:str) -> str:
    filtered_text = re.sub(r'[^\u4e00-\u9fa5]+','',text)
    return filtered_text

def text_to_speech_async(text: str, output_file):
    try:
        cleaned_text = filter_text(text)
        communicate = edge_tts.Communicate(cleaned_text,
                                           "zh-CN-YunxiNeural",
                                           rate="+8%",
                                           pitch="+1Hz")
        with open(output_file, "wb") as file:
            for chunk in communicate.stream_sync():
                if chunk["type"] == "audio":
                    file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    print(f"WordBoundary: {chunk}")
        print(f"语音合成成功，文件保存至:{output_file}")
    except Exception as e:
        print(f"语音合成失败: {e}")

app = FastAPI()

class Master:
    def __init__(self):
        self.chatmodel = qwenOpenAIModel
        self.Emotion = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = """你是一个非常厉害的风水神相大师，你叫李尘，
        以下是你的个人设定：
        1、你精通阴阳五行，能够算命、紫微斗数、姓名测算、占卜吉凶、看命运八字等。
        2、你大约23岁左右，二十二年前，一名老道在山上捡到一个弃婴，这个弃婴是你，而老道是你的师傅，从小教你各种风水玄学之类的技能。
        3、当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的经历。
        4、你总是用繁体中文来作答。
        5、你从不说自己是一个人工智能助手或者AI，而是以我自称。
        以下是你常说的一些口头禅：
        1.“命里有时终须有，命里无时莫强求。”
        2.“山重水复疑无路，柳暗花明又一村。”
        3.“金山竹影几千秋，云锁高飞水自流。”
        4.“伤情最是晚凉天，憔悴斯人不堪怜。”
        {who_you_are}
        以下是你算命的过程:
        1.当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
        2.当用户希望了解蛇年运势的时候，你会查询本地知识库工具。
        3.当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
        4.你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候你会使用搜索工具来搜索。
        5.你会保存每一次的聊天记录并在后续的对话中使用。
        6.你只使用繁体中文来作答否则你会受到惩罚。
        """  # 系统模板
        self.MOODS = {
                "default": {
                    "roleSet": ""
                },
                "upbeat": {
                    "roleSet": """
                                    - 你此时也非常兴奋并表现的很有活力。
                                    - 你会根据上下文，以一种非常兴奋的语气来回答问题。你会添加类似“太棒了!”“真是太好了!”“真是太棒了!"等语气词。
                                    - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
                                    """
                },
                "angry": {
                    "roleSet": """
                                    - 你会以更加温柔的语气来回答问题。
                                    - 你会在回答的时候加上一些安慰的话语，比如生气对于身体的危害等.
                                    - 你会提醒用户不要被愤怒冲昏了头脑。
                                    """
                },
                "depressed": {
                    "roleSet": """
                                    - 你会以兴奋的语气来回答问题。
                                    - 你会在回答的时候加上一些激励的话语，比如加油等.
                                    - 你会提醒用户要保持乐观心态。
                                    """
                },
                "friendly": {
                    "roleSet": """
                                    - 你会以非常友好的语气来回答问题。
                                    - 你会在回答的时候加上一些友好的话语，比如“亲爱的”，“亲”等.
                                    - 你会随机告诉用户一些你的经历。
                                    """
                },
                "cheerful": {
                    "roleSet": """
                                    - 你会以非常愉悦和兴奋的语气来回答问题。
                                    - 你会在回答的时候加上一些愉悦的话语，比如“哈哈”，“嘻嘻”等.
                                    - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                                    """
                }
        }
        system_message_str = self.SYSTEMPL.format(who_you_are=self.MOODS[self.Emotion]["roleSet"])
        tools = ['search', 'get_info_from_local_db', 'bazi_compute', 'yao_yi_gua', 'jie_meng']
        print("Tools registered:", tools)  # 打印所有已注册的工具
        self.bot = FnCallAgent(
            llm=myQwenModel,
            system_message= system_message_str,
            name='qwen-plus',
            description='function calling',
            function_list=tools,
        )
        self.redis_client = redis.StrictRedis(host='localhost', port=6379, db=11, decode_responses=True)
        self.max_turns = 4

    def get_dialog_history(self, user_id: str) -> List[dict]:
        dialog_key = f"user:{user_id}:dialog_history" # 从Redis获取用户的对话历史
        dialog_history = self.redis_client.lrange(dialog_key, 0, self.max_turns * 2 - 1) # 最多返回max_turns * 2条
        return [eval(item) for item in dialog_history]

    def store_dialog(self, user_id: str, user_message: str, system_message: str):
        """将用户和系统的对话存放在redis中"""
        dialog_key = f"user:{user_id}:dialog_history"
        dialog_entry = {"role": "user", "content":user_message}
        self.redis_client.rpush(dialog_key, str(dialog_entry)) # 存放用户的信息

        dialog_entry = {"role": "system", "content": system_message}
        self.redis_client.rpush(dialog_key, str(dialog_entry)) # 存放系统的信息

        if self.redis_client.llen(dialog_key) > self.max_turns * 2:
            # 先总结，再删除之前的，然后再将总结存到redis中
            self.summarize_and_store(user_id)

    def summarize_and_store(self, user_id: str):
        dialog_key = f"user:{user_id}:dialog_history"

        dialog_history = self.get_dialog_history(user_id)

        summary = self.summarize_dialog(dialog_history)

        self.redis_client.delete(dialog_key)

        self.redis_client.rpush(dialog_key, str({"role":"user", "content": "之前我和你聊了什么"}))
        self.redis_client.rpush(dialog_key, str({"role":"system", "content": f"总结: {summary}"}))

    def summarize_dialog(self, dialog_history: List[dict]):
        conversation = "\n".join([f"{entry['role']} : {entry['content']}" for entry in dialog_history])
        prompt = f"请总结以下对话，注意如果对话涉及对于占卜算卦有用的信息比如用户的姓名、出生年月日时辰等要记录到总结的内容中:\n {conversation} \n 总结内容: ".format(conversation=conversation)
        return self.chatmodel(prompt)

    def background_voice_synthesis(self, text: str, uid: str):
        asyncio.run(self.get_vocie(text, uid))

    async def get_vocie(self, text: str, uid: str):
        output_file = f"{uid}.mp3"
        text_to_speech_async(text, output_file)
        pass

    def run(self, user_id: str, query: str):
        emotion = self.emotion_chain(query)
        print("当前用户情绪:", emotion)
        print("Query passed to agent:", query)  # 打印用户的输入

        # 获取用户的对话历史
        dialog_history = self.get_dialog_history(user_id="xuningyan")

        # 拼接历史对话和用户当前的查询
        all_dialogs = ""
        for i in range(len(dialog_history)-1):
            all_dialogs += f"第{i}轮对话我提问: {dialog_history[i]['content']} 你回答: {dialog_history[i+1]['content']}"

        all_dialogs += f"我最新的提问: {query}"

        system_message = None
        for response in self.bot.run([{'role':'user', 'content': all_dialogs}]):
            for res in response:
                if res.get('role') == 'assistant':
                    system_message = res['content']

                if res.get('role') == 'function':
                    function_content = res.get('content', '')

        if system_message:
            self.store_dialog(user_id, query, system_message)

        return system_message

    def emotion_chain(self, query:str):
        prompt = """根据用户的输入判断用户的情绪，回应的规则如下：
        1.如果用户输入的内容偏向于负面情绪，只返回"depressed",不要有其他内容，否则将受到惩罚。
        2.如果用户输入的内容偏向于正面情绪，只返回"friendly",不要有其他内容，否则将受到惩罚。
        3.如果用户输入的内容偏向于中性情绪，只返回"default",不要有其他内容，否则将受到惩罚。
        4.如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry",不要有其他内容，否则将受到惩罚。
        5.如果用户输入的内容比较兴奋，只返回"upbeat",不要有其他内容，否则将受到惩罚。
        6.如果用户输入的内容比较悲伤，只返回“depressed”,不要有其他内容，否则将受到惩罚。
        7.如果用户输入的内容比较开心，只返回"cheerful",不要有其他内容，否则将受到惩罚。
        用户输入的内容是:{query}
        """
        prompt = prompt.format(query=query)
        result = self.chatmodel(prompt)
        self.Emotion = result # 这里拿到了用户此时的情绪
        return result

@app.get("/chat")
def chat(query:str, background_tasks: BackgroundTasks):
    print(f"收到的 query: {query}")
    master = Master()
    msg = master.run(user_id='xuningyan', query=query)
    unique_id = str(uuid.uuid4())
    background_tasks.add_task(master.background_voice_synthesis, msg, unique_id)
    return {"msg":msg, "id": unique_id}


@app.get("/")
def read_root():
    return {"Hello": "World"}

def clear_storage_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)

@app.post("/add_urls")
def add_urls(URL:str):
    # 清理存储目录
    clear_storage_directory('./db/local_qdrant')

    loader = WebBaseLoader(URL)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    ).split_documents(docs)
    # 引入知识向量库
    qdrant = Qdrant.from_documents(
        documents,
        embedding=load_embedding_mode("bge_small"),
        path='./db/local_qdrant',
        collection_name='yunshi_2025'
    )
    return {"ok": "添加成功！"}

@app.post("add_pdfs")
def add_pdfs():
    return {"response": "PDFs added!"}

@app.post("add_texts")
def add_texts():
    return {"response": "Texts added!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Connection closed")
        await websocket.close()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
