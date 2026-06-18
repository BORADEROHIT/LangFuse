from dotenv import load_dotenv
load_dotenv()

from langfuse import observe
from langfuse import get_client

langfuse = get_client()  # Reads credentials from ENV

@observe(name="my_function")
def my_function(a, b):
    return a + b

# Call with a specific trace context
my_function(2, 2)
