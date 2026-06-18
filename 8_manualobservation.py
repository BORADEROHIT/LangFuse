from langfuse import get_client
 
langfuse = get_client()
 
span = langfuse.start_observation(name="manual-span")
span.update(input="Data for side task")
child = span.start_observation(name="child-span", as_type="generation")
child.end()
span.end()