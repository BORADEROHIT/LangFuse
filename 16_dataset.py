from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

# Initialize Langfuse client
langfuse = Langfuse()

langfuse.create_dataset(
    name="demo",
    # optional description
    description="My first dataset",
    # optional metadata
    metadata={
        "author": "Anu",
        "date": "2026-01-06",
        "type": "benchmark"
    }
)

langfuse.create_dataset_item(
    dataset_name="demo",
    # any python object or value, optional
    input={
        "text": "hello world"
    },
    # any python object or value, optional
    expected_output={
        "text": "hello world"
    },
)