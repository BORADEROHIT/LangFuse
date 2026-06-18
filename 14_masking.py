'''Define a masking function that uses a regular expression to detect and replace credit card numbers.
Configure the masking function in Langfuse.
Create a sample function to simulate processing sensitive data.
Observe the trace to see the masked output.'''
from dotenv import load_dotenv
load_dotenv()


import re
from langfuse import Langfuse, observe, get_client
 
# Step 2: Define the masking function
def masking_function(data, **kwargs):
    if isinstance(data, str):
        # Regular expression to match credit card numbers (Visa, MasterCard, AmEx, etc.)
        pattern = r'\b(?:\d[ -]*?){13,19}\b'
        data = re.sub(pattern, '[REDACTED CREDIT CARD]', data)
    return data
 
# Step 3: Configure the masking function
langfuse = Langfuse(mask=masking_function)
 
# Step 4: Create a sample function with sensitive data
@observe()
def process_payment():
    # Simulated sensitive data containing a credit card number
    transaction_info = "Customer paid with card number 4111 1111 1111 1111."
    return transaction_info
 
# Step 5: Observe the trace
result = process_payment()
 
print(result)
# Output: Customer paid with card number [REDACTED CREDIT CARD].
 
# Flush events in short-lived applications
langfuse.flush()