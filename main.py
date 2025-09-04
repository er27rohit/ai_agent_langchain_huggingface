from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
print("Token", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

llm = HuggingFaceEndpoint(
    # repo_id="HuggingFaceH4/zephyr-7b-beta", # Working, need API TOKEN
    # repo_id="mradermacher/oh-dcft-v3.1-claude-3-5-sonnet-20241022-GGUF", # Not working
    repo_id="openai/gpt-oss-120b", # Working with Token and Fast
    # repo_id="deepseek-ai/DeepSeek-R1-0528",  # Working and need API TOKEN, shows thinking process as well
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)
try:
    response = chat_model.invoke("What is the capital of France? Reply in one word")
    print(response.content)
except Exception as e:
    print("Error:", e)