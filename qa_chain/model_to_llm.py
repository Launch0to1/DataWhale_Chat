from langchain_community.llms import VLLM
def model_to_llm(model:str=None, temperature:float=0.0, appid:str=None, api_key:str=None,Spark_api_secret:str=None,Wenxin_secret_key:str=None):
        llm = VLLM(model=model)  
        return llm