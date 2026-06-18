'''Build a support chatbot in LangGraph that can answer common questions
Tracing the chatbot input and output using Langfuse'''
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from IPython.display import Image, display
 
class State(TypedDict):
    # Messages have the type "list". 
    # add_messages appends messages to the list, rather than overwriting them
    messages: Annotated[list, add_messages]
 
graph_builder = StateGraph(State)
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()
 
llm = ChatOllama(model="llama3.1",temperature=0.3)
 
# The chatbot node function takes the current State as input and returns an updated messages list. 
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
 
# Add a "chatbot" node. Nodes represent units of work. They are typically regular python functions.
graph_builder.add_node("chatbot", chatbot)
 
# Add an entry point. This tells our graph where to start its work each time we run it.
graph_builder.set_entry_point("chatbot")
 
# Set a finish point. This instructs the graph "any time this node is run, you can exit."
graph_builder.set_finish_point("chatbot")
 
# To be able to run our graph, call "compile()" on the graph builder. This creates a "CompiledGraph" we can use invoke on our state.
graph = graph_builder.compile()

for s in graph.stream({"messages": [HumanMessage(content = "What is self love, answer in a sentence?")]},
                      config={"callbacks": [langfuse_handler]}):
    print(s)

#display(Image(graph.get_graph().draw_mermaid_png()))