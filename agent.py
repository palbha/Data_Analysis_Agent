import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def get_agent(df: pd.DataFrame):
    """
    Creates and returns a Pandas DataFrame Agent using a Hugging Face model
    and configures Langfuse tracing.
    """
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_token or hf_api_token == "your_huggingface_token_here":
        raise ValueError("Please set a valid HUGGINGFACEHUB_API_TOKEN in your .env file")
    
    # Initialize Langfuse handler for tracing
    # The handler automatically picks up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from environment
    langfuse_handler = CallbackHandler()

    # Use Hugging Face's OpenAI-compatible API for better compatibility with agents
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key=hf_api_token,
        base_url="https://router.huggingface.co/v1",
        max_tokens=512,
        temperature=0.1,
    )

    # Create the agent that can execute python code to answer questions about the dataframe
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="openai-tools",
        allow_dangerous_code=True # Note: This allows arbitrary code execution. Ensure you trust the environment.
    )
    
    return agent, langfuse_handler

def run_query(agent, langfuse_handler, query: str):
    """
    Runs a query through the agent with Langfuse tracing enabled.
    """
    # The callback handler ensures the trace is sent to Langfuse
    response = agent.invoke(
        {"input": query},
        config={"callbacks": [langfuse_handler]}
    )
    return response["output"]
