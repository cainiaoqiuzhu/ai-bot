# encoding:utf-8
from langchain.embeddings import HuggingFaceEmbeddings


# 加载embedding
embedding_model_dict = {
    "ernie_tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie_base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing424/text2vec-base-chinese",
    "bge_small": "D:/HuggingFaceModel/bge-small-zh-v1.5"
}

def load_embedding_mode(model_name="ernie-tiny"):
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name = embedding_model_dict[model_name],
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )