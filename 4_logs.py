from dotenv import load_dotenv
load_dotenv()
from langfuse import observe, get_client
langfuse = get_client() 
 
@observe()
def my_function():
    langfuse = get_client()
    langfuse.update_current_span(
        level="WARNING",
        status_message="This is a warning"
    )
my_function()