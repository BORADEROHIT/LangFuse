from dotenv import load_dotenv
load_dotenv()
"""
This script demonstrates how to integrate Langfuse tracing with a fake/dummy LangChain LLM.

It serves as a minimal example to verify the Langfuse integration without incurring costs or requiring a real LLM provider.
The script:
1. Configures a `FakeListLLM` from LangChain to return predefined responses.
2. Initializes the Langfuse client and LangChain callback handler.
3. Invokes the dummy LLM with the callback handler attached, allowing Langfuse to trace the execution.
"""

from langchain_community.llms.fake import FakeListLLM
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()  # Reads credentials from ENV

dummy_llm = FakeListLLM(responses = ["dummy dummy dummy"])

langfuse_handler = CallbackHandler() ## Enables tracing for LangChain/LangGraph

print("Invoking LangChain Dummy LLM...")
response1 = dummy_llm.invoke("Are you real?", config={"callbacks": [langfuse_handler]})
print(f"Response 1: {response1}")