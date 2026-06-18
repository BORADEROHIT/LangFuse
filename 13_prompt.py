

from langfuse import Langfuse

# Initialize Langfuse client
langfuse = Langfuse()

# Get production prompt
#prompt = langfuse.get_prompt("guide-india")

# Get by label
# You can use as many labels as you'd like to identify different deployment targets
prompt = langfuse.get_prompt("guide-india", label="test")
#print(prompt.prompt)
#prompt = langfuse.get_prompt("guide-india", label="production")
print(prompt.prompt)
final_prompt=prompt.compile(query="hi")
print(final_prompt)
# Get by version number, usually not recommended as it requires code changes to deploy new prompt versions
#langfuse.get_prompt("guide-india", version=1)
