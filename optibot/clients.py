from openai import OpenAI

def create_openai_client(api_key):
    client = OpenAI(api_key=api_key)
    return client