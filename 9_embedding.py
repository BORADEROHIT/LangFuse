from langchain_ollama import OllamaEmbeddings
from langfuse import get_client

langfuse = get_client()  # Reads credentials from ENV
emb = OllamaEmbeddings(model="llama3.1")

text = "Langfuse provides observability, prompt management, and evaluation for LLM apps."

with langfuse.start_as_current_observation(as_type="span", name="ollama-embedding") as span:
    # Attach the INPUT so it shows in the dashboard
    span.update(input={"value": text})            # <- important

    # Perform the embedding
    vector = emb.embed_query(text)

    # Attach the OUTPUT (you can store metadata rather than full vector)
    span.update(output={"dim": len(vector),"preview_first_16": vector[:16],})      # store dimension or a preview
    # If you want to preview a few numbers only:
    # span.update(output={"preview": vector[:8]})

    # Optional extra metadata
    span.update(metadata={"embedding_model": "ollama:llama3.1"})

# Flush events so they appear in Langfuse dashboard
langfuse.flush()

print(f"Embedding generated! Vector length: {len(vector)}")
