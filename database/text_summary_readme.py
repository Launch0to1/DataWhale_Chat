import os
from dotenv import load_dotenv
import openai
from test_get_all_repo import get_repos
from bs4 import BeautifulSoup
import markdown
import re
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
# Load environment variables
load_dotenv()
TOKEN = os.getenv('TOKEN')

# 过滤文本中链接防止大语言模型风控
def remove_urls(text):
    # 正则表达式模式，用于匹配URL
    url_pattern = re.compile(r'https?://[^\s]*')
    # 替换所有匹配的URL为空字符串
    text = re.sub(url_pattern, '', text)
    # 正则表达式模式，用于匹配特定的文本
    specific_text_pattern = re.compile(r'扫描下方二维码关注公众号|提取码|关注|科学上网|回复关键词|侵权|版权|致谢|引用|LICENSE'
                                       r'|组队打卡|任务打卡|组队学习的那些事|学习周期|开源内容|打卡|组队学习|链接')
    # 替换所有匹配的特定文本为空字符串
    text = re.sub(specific_text_pattern, '', text)
    return text

# 抽取md中的文本
def extract_text_from_md(md_content):
    # Convert Markdown to HTML
    html = markdown.markdown(md_content)
    # Use BeautifulSoup to extract text
    soup = BeautifulSoup(html, 'html.parser')

    return remove_urls(soup.get_text())

def generate_llm_summary(repo_name, readme_content,llm,sampling_params,tokenizer):
    prompt = f"1：这个仓库名是 {repo_name}. 此仓库的readme全部内容是: {readme_content}\
               2:请用约200以内的中文概括这个仓库readme的内容,返回的概括格式要求：这个仓库名是...,这仓库内容主要是..."
    
    # 具体调用
    messages = [{"role": "system", "content": "你是一个人工智能助手"},
                {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = llm.generate(text, sampling_params)
    #print(outputs)
    output = outputs[0]
    prompt = output.prompt
    generated_text = output.outputs[0].text
    if r"</think>" in generated_text:
        think_content, answer_content = generated_text.split(r"</think>")
    else:
        think_content = ""
        answer_content = generated_text
    #print(f"Prompt: {prompt!r}, Think: {think_content!r}, Answer: {answer_content!r}")
    
    return answer_content

def get_llm( model, tokenizer=None, max_tokens=8192, temperature=0.6, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer,quantization='fp8', max_model_len=max_model_len,trust_remote_code=True)
    #outputs = llm.generate(prompts, sampling_params)
    return llm,sampling_params

def main(org_name,export_dir,summary_dir,llm,sampling_params,tokenizer):
    # 获取仓库列表
    repos = get_repos(org_name, TOKEN, export_dir)

    # Create a directory to save summaries
    os.makedirs(summary_dir, exist_ok=True)

    for id, repo in enumerate(repos):
        repo_name = repo['name']
        readme_path = os.path.join(export_dir, repo_name, 'README.md')
        print(repo_name)
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as file:
                readme_content = file.read()
            # Extract text from the README
            readme_text = extract_text_from_md(readme_content)
            # Generate a summary for the README
            # 访问受限，每min一次
            time.sleep(60)
            print('第' + str(id) + '条' + 'summary开始')
            try:
                summary = generate_llm_summary(repo_name, readme_text,llm,sampling_params,tokenizer)
                print(summary)
                # Write summary to a Markdown file in the summary directory
                summary_file_path = os.path.join(summary_dir, f"{repo_name}_summary.md")
                with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                    summary_file.write(f"# {repo_name} Summary\n\n")
                    summary_file.write(summary)
            except openai.OpenAIError as e:
                summary_file_path = os.path.join(summary_dir, f"{repo_name}_summary风控.md")
                with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                    summary_file.write(f"# {repo_name} Summary风控\n\n")
                    summary_file.write("README内容风控。\n")
                print(f"Error generating summary for {repo_name}: {e}")
                # print(readme_text)
        else:
            print(f"文件不存在: {readme_path}")
            # If README doesn't exist, create an empty Markdown file
            summary_file_path = os.path.join(summary_dir, f"{repo_name}_summary不存在.md")
            with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                summary_file.write(f"# {repo_name} Summary不存在\n\n")
                summary_file.write("README文件不存在。\n")
if __name__ == '__main__':
    # 配置组织名称
    org_name = 'datawhalechina'
    # 配置 export_dir
    export_dir = "database/readme_db"  # 请替换为实际readme的目录路径
    summary_dir="knowledge_db/readme_summary"# 请替换为实际readme的概括的目录路径
    model="/data/xmx/model/DeepSeek-R1-Distill-Qwen-7B"  #deepseek-chat,gpt-3.5-turbo,moonshot-v1-8k
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    llm,sampling_params = get_llm(model)
    main(org_name,export_dir,summary_dir,llm,sampling_params,tokenizer)
