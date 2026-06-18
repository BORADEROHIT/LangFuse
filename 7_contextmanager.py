from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

langfuse = get_client()  # Reads credentials from ENV
from langchain_ollama import ChatOllama

import warnings
warnings.filterwarnings("ignore", category=Warning)
langfuse_handler = CallbackHandler()
# Initialize ChatOllama
# -------------------------------
llm = ChatOllama(
    model="llama3.1",
    temperature=0.3,
    callbacks=[langfuse_handler]  # Attach Langfuse callback
)
 
with langfuse.start_as_current_observation(
    as_type="span",
    name="user-request-pipeline",
    input={"user_query": "Tell me a joke"},
) as root_span:
    with propagate_attributes(user_id="user_123", session_id="session_abc"):
        response = llm.invoke(
            [
                ("system", "You are a witty comedian."),
                ("user", "Tell me a joke"),
            ],
            config={"callbacks": [langfuse_handler]},  # critical line
        )

        # Optionally update the generation child manually too
        root_span.update(output={"final_joke": response.content})

        '''with langfuse.start_as_current_observation(
            as_type="generation",
            name="joke-generation",
            model="llama3.1",
        ) as generation:
            generation.update(output="Why did the span cross the road?")
 
    root_span.update(output={"final_joke": "..."})
    '''
        