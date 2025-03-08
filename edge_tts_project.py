import edge_tts


# VOICE = "en-GB-SoniaNeural"
# 晓晓 女
# VOICE = "zh-CN-XiaoxiaoNeural"
# 云健 -不好听
#  VOICE = "zh-CN-YunjianNeural"
# 云溪 男好听
VOICE = "zh-CN-YunxiNeural"
# 云夏 女好听
# VOICE = "zh-CN-YunxiaNeural"
# 云扬 男官方版本
# VOICE = "zh-CN-YunyangNeural"
# 辽宁 小北女
# VOICE = "zh-CN-liaoning-XiaobeiNeural"
# 台湾 女
# VOICE = "zh-TW-HsiaoChenNeural"
# 台湾男
# VOICE = "zh-TW-YunJheNeural"
# 小云 台湾 女
# VOICE = "zh-TW-HsiaoYuNeural"
# 陕西方言 小妮
# VOICE = "zh-CN-shaanxi-XiaoniNeural"
# 女 晓意
# VOICE = "zh-CN-XiaoyiNeural"
# 女 香港
# VOICE ="zh-HK-HiuGaaiNeural"
# 女 香港
# VOICE = "zh-HK-HiuMaanNeural"
# 男 香港
# VOICE = "zh-HK-WanLungNeural"


def text_to_speech(text: str, output_file):
    communicate = edge_tts.Communicate(text, VOICE)
    with open(output_file, "wb") as file:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                print(f"WordBoundary: {chunk}")


if __name__ == "__main__":
    text_to_speech("你好，你叫什么名字", "output_file.mp3")