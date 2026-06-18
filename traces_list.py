from langfuse import observe #observe decorator wraps your function so Langfuse can automatically log/trace calls
from langfuse import get_client

langfuse = get_client() 
'''
#Fetch single trace
trace = langfuse.api.trace.get("db69ff1b81ce3555e7a576b2db4cffb6") #trace id
print(trace)

#Fetch multiple observations
observations = langfuse.api.observations.get_many(limit=2)
print(observations)
'''

#Fetch multiple sessions
sessions = langfuse.api.sessions.list(limit=3)
print(sessions)
'''
#Fetch single observation
observation = langfuse.api.legacy.observations_v1.get("26702fa734f7e4a8")
print(observation)


#Fetch multiple observations
observations = langfuse.api.observations.get_many(limit=2)
print(observations)

#Fetch multiple sessions
sessions = langfuse.api.sessions.list(limit=50)
print(sessions)

'''