import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    if embedding == 'm3e':
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    elif embedding == 'bge-m3':
        return HuggingFaceEmbeddings(model_name="/data/xmx/model/bge-m3")
    elif embedding == 'gte_Qwen2-7B-instruct':
        return HuggingFaceEmbeddings(model_name="T/data/xmx/model/gte_Qwen2-7B-instruct")
    else:
        raise ValueError(f"embedding {embedding} not support ")
