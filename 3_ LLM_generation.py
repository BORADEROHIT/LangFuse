from dotenv import load_dotenv
load_dotenv()
"""
This script demonstrates how to integrate Langfuse tracing with a LLM generation.
"""
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()  # Reads credentials from ENV
from langchain_ollama import ChatOllama

from langchain_aws import ChatBedrock
import warnings
warnings.filterwarnings("ignore", category=Warning)
langfuse_handler = CallbackHandler()
# Initialize ChatOllama
# -------------------------------
chat = ChatOllama(
    model="llama3.1",
    temperature=0.3,
    callbacks=[langfuse_handler]  # Attach Langfuse callback
)

response = chat.invoke(
    "What is Jupiter retrograde in astrology in 15 words?"
)

print("\nModel Response:")
print(response.content)