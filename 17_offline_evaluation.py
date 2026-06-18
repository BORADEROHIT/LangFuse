'''Evaluating LangGraph agents'''
from dotenv import load_dotenv
load_dotenv()


import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrock
from langfuse import get_client
from langgraph.graph.message import add_messages
from typing import Annotated


class EmailState(TypedDict):
    email: Dict[str, Any]           
    is_spam: Optional[bool]         
    spam_reason: Optional[str]      
    email_category: Optional[str]   
    draft_response: Optional[str]   
    messages: List[Dict[str, Any]] 

model= ChatBedrock(
    model_id="amazon.nova-lite-v1:0",  
    region_name="us-east-1",
    temperature=0)
# Define nodes
def read_email(state: EmailState):
    email = state["email"]
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")
    return {}
 
def classify_email(state: EmailState):
    email = state["email"]
    
    prompt = f"""
As Alfred the butler of Mr wayne and it's SECRET identity Batman, analyze this email and determine if it is spam or legitimate and should be brought to Mr wayne's attention.
 
Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}
 
First, determine if this email is spam.
answer with SPAM or HAM if it's legitimate. Only return the answer
Answer :
    """
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    response_text = response.content.lower()
    print(response_text)
    is_spam = "spam" in response_text and "ham" not in response_text
    
    if not is_spam:
        new_messages = state.get("messages", []) + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]
    else :
        new_messages = state.get("messages", [])
    
    return {
        "is_spam": is_spam,
        "messages": new_messages
    }
 
def handle_spam(state: EmailState):
    print(f"Alfred has marked the email as spam.")
    print("The email has been moved to the spam folder.")
    return {}
 
def drafting_response(state: EmailState):
    email = state["email"]
    
    prompt = f"""
As Alfred the butler, draft a polite preliminary response to this email.
 
Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}
 
Draft a brief, professional response that Mr. Wayne can review and personalize before sending.
    """
    
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    return {
        "draft_response": response.content,
        "messages": new_messages
    }
 
def notify_mr_wayne(state: EmailState):
    email = state["email"]
    
    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["draft_response"])
    print("="*50 + "\n")
    
    return {}
# Define routing logic
def route_email(state: EmailState) -> str:
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"
    
# Create the graph
email_graph = StateGraph(EmailState)
 
# Add nodes
email_graph.add_node("read_email", read_email) # the read_email node executes the read_mail function
email_graph.add_node("classify_email", classify_email) # the classify_email node will execute the classify_email function
email_graph.add_node("handle_spam", handle_spam)  
email_graph.add_node("drafting_response", drafting_response) 
email_graph.add_node("notify_mr_wayne", notify_mr_wayne) 

# Add edges
email_graph.add_edge(START, "read_email") # After starting we go to the "read_email" node
 
email_graph.add_edge("read_email", "classify_email") # after_reading we classify
 
# Add conditional edges
email_graph.add_conditional_edges(
    "classify_email", # after classify, we run the "route_email" function"
    route_email,
    {
        "spam": "handle_spam", # if it return "Spam", we go the "handle_span" node
        "legitimate": "drafting_response" # and if it's legitimate, we go to the "drafting response" node
    }
)
 
# Add final edges
email_graph.add_edge("handle_spam", END) # after handling spam we always end
email_graph.add_edge("drafting_response", "notify_mr_wayne")
email_graph.add_edge("notify_mr_wayne", END) # after notifyinf Me wayne, we can end  too

# Compile the graph
compiled_graph = email_graph.compile()

# Example emails for testing
legitimate_email = {
    "sender": "Joker",
    "subject": "The new AI era ! ",
    "body": "Mr. Wayne,I am reaching out to share how our new AI-driven approach can enhance efficiency and transform the way we work."
}
 
spam_email = {
    "sender": "Crypto bro",
    "subject": "The best investment of 2025",
    "body": "Mr Wayne, I just launched an ALT coin and want you to buy some !"
}

from langfuse.langchain import CallbackHandler
 
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()
 
# Process legitimate email
print("\nProcessing legitimate email...")
legitimate_result = compiled_graph.invoke(
    input = {
        "email": legitimate_email,
        "is_spam": None,
        "draft_response": None,
        "messages": []
        },
    config={"callbacks": [langfuse_handler]}
)
 
# Process spam email
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke(
    input = {
        "email": spam_email,
        "is_spam": None,
        "draft_response": None,
        "messages": []
        },
    config={"callbacks": [langfuse_handler]}
) 

#In offline evaluation, you typically have a benchmark dataset (with prompt and expected output pairs)
import pandas as pd
from datasets import load_dataset
 
# Fetch search-dataset from Hugging Face
dataset = load_dataset("junzhang1207/search-dataset", split = "train")
df = pd.DataFrame(dataset)
print("First few rows of search-dataset:")
print(df.head())

#Next, we create a dataset entity in Langfuse to track the runs. Then, we add each item from the dataset to the system.

from langfuse import Langfuse
langfuse = Langfuse()
 
langfuse_dataset_name = "qa-dataset_langgraph-agent"
 
# Create a dataset in Langfuse
langfuse.create_dataset(
    name=langfuse_dataset_name,
    description="q&a dataset uploaded from Hugging Face",
    metadata={
        "date": "2025-03-21",
        "type": "benchmark"
    }
)

df_10 = df.sample(10) # For this example, we upload only 10 dataset questions
 
for idx, row in df_10.iterrows():
    langfuse.create_dataset_item(
        dataset_name=langfuse_dataset_name,
        input={"text": row["question"]},
        expected_output={"text": row["expected_answer"]}
    )

#define a task function my_task() that wraps our LangGraph agent.
# Define the task function we pass to the experiment runner method
def my_task(*, item, **kwargs):
    
    question = item.input["text"]
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
 
    graph_builder = StateGraph(State)
 
    def chatbot(state: State):
        return {"messages": [model.invoke(state["messages"])]}
    
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.set_entry_point("chatbot")
    graph_builder.set_finish_point("chatbot")
    graph = graph_builder.compile()
 
    response = graph.invoke(
        input={"messages": [HumanMessage(content=question)]},
        config={"callbacks": [langfuse_handler]}
        )
        
    print(question)
    print(response["messages"][1].content)
 
    return response["messages"][1].content

#Finally, we use the experiment runner SDK to run our task function against each dataset item. The experiment runner handles concurrent execution, automatic tracing, and evaluation.
# Fetch dataset and run experiment
dataset = langfuse.get_dataset('qa-dataset_langgraph-agent')
 
result = dataset.run_experiment(
    name="run_ChatBedrock",
    description="My first run",
    task=my_task,
    metadata={"model": "ChatBedrock"}
)
 
# Flush the langfuse client to ensure all data is sent to the server at the end of the experiment run
langfuse.flush()
print(result.format())