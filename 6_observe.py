from langfuse import observe
import asyncio
 
@observe()
def my_data_processing_function(data, parameter):
    return {"processed_data": data, "status": "ok"}
 
@observe(name="llm-call", as_type="generation")
async def my_async_llm_call(prompt_text):
    return "LLM response"

my_data_processing_function("sample-data", 123)

asyncio.run(
    my_async_llm_call("Explain Langfuse tracing")
)