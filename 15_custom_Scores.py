from dotenv import load_dotenv
load_dotenv()


from langfuse import get_client
langfuse = get_client()
 
langfuse.create_score(
    name="correctness",
    value=0.9,
    trace_id="ab1f8bae794829e9bf241cb4e6722862",
    #observation_id="observation_id_here", # optional
    #session_id="session_id_here", # optional, Id of the session the score relates to
    data_type="NUMERIC", # optional, inferred if not provided
    comment="Factually correct", # optional
)