from langfuse import observe, propagate_attributes
from langfuse import get_client
from langchain_ollama import ChatOllama
langfuse = get_client()  # Reads credentials from ENV
llm = ChatOllama(
    model="llama3.1",  # Attach Langfuse callback
)
def llm_generate_with_ollama(prompt_text: str) -> str:
    """
    A child 'generation' observation using Ollama's Chat model via LangChain.
    The decorator captures input (prompt_text) and output automatically.
    """
    messages = [("system", "You are a helpful guide for travelers in India."),
                ("user", prompt_text)]
    response = llm.invoke(messages)
    return response.content

@observe(name="process_request")
def process_request(query):
    with propagate_attributes(user_id="abraKaDabra",session_id="test-session-001"):
        # All nested observations automatically inherit session_id
        
        # prompt = langfuse.get_prompt("guide-india")
        prompt = langfuse.get_prompt("guide-india", label="latest")
        # print(prompt.prompt)
        final_prompt = prompt.compile(query=query)
        print(final_prompt)
        res = llm_generate_with_ollama(final_prompt)
        return res
    
process_request("Hello, can you guide me through India?")