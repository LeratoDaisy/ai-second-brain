import os
from openai import OpenAI

# Read API key from environment
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)