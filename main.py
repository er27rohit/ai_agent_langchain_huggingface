from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import save_tool, search_tool, wiki_tool
import os

load_dotenv()

llm_var = "huggingface"  # Change to "openai" or "anthropic" to use different models

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Your response must strictly follow this JSON format and provide no other text:
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

if (llm_var == "huggingface"):
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    print("Using Hugging Face model")
    # print("HuggingFace Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    chat_model = ChatHuggingFace(llm=HuggingFaceEndpoint(
        # repo_id="HuggingFaceH4/zephyr-7b-beta", # Working, need API TOKEN
        # repo_id="mradermacher/oh-dcft-v3.1-claude-3-5-sonnet-20241022-GGUF", # Not working
        # repo_id="openai/gpt-oss-120b", # Working with Token and Fast
        repo_id="deepseek-ai/DeepSeek-R1-0528",  # Working and need API TOKEN, shows thinking process as well
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",  # let Hugging Face choose the best provider for you
    ))

    # tools = [
    # Tool(
    #     name="dummy_tool",
    #     func=lambda query: "This is a dummy tool response.",
    #     description="A dummy tool for testing purposes."
    # )
    # ]
    tools = [wiki_tool, save_tool]
    agent = create_tool_calling_agent(
        llm=chat_model,
        prompt=prompt,
        tools=tools,
    )
    # print(prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    raw_response = agent_executor.invoke({"query": "What is the capital of France? Save the file"})
    # raw_response = {'query': 'What is the capital of France?', 'output': ' {\n  "topic": "Capital of France",\n  "summary": "The capital of France is Paris.",\n  "sources": [\n    "https://en.wikipedia.org/wiki/Paris"\n  ],\n  "tools_used": []\n}'}

    # raw_response = {'query': 'What is the capital of France?', 'output': 'commentary{\n  "topic": "Capital of France",\n  "summary": "Paris is the capital and largest city of France, serving as the country\'s political, economic, and cultural center.",\n  "sources": [\n    "https://en.wikipedia.org/wiki/Paris"\n  ],\n  "tools_used": [\n    "functions.wikipedia"\n  ]\n}'}
    if "output" in raw_response:
        raw_response["output"] = "{" + raw_response["output"].split("{", 1)[-1]

    # print(raw_response)
    # quit()

    try:
        structured_response = parser.parse(raw_response.get("output", ""))
        # structured_response = raw_response.get("output", "")

        print("\n\n\nStructured Response:\n", structured_response)
    except Exception as e: 
        print("Error parsing response:", e) 

    # try:
    #     response = chat_model.invoke("What is the capital of France? Reply in one word")
    #     print(response.content)
    # except Exception as e:
    #     print("Error:", e)

elif (llm_var == "openai"):
    from langchain_openai import ChatOpenAI  # Uncomment this line and comment the above line to use OpenAI's GPT models
    print("Using OpenAI model")
    # print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))

    chat_model = ChatOpenAI(model="gpt-4", temperature=0)
    try:
        response = chat_model.invoke("What is the capital of France? Reply in one word")
        print(response.content)
    except Exception as e:
        print("Error:", e)

elif (llm_var == "anthropic"):
    from langchain_anthropic import ChatAnthropic
    print("Using Anthropic model")
    # print("Anthropic API Key:", os.getenv("ANTHROPIC_API_KEY"))
    
    chat_model = ChatAnthropic(model="claude-2", temperature=0)
    try:
        response = chat_model.invoke("What is the capital of France? Reply in one word")
        print(response.content)
    except Exception as e:
        print("Error:", e)


