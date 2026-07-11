import os

from dotenv import load_dotenv
from ollama import Client

load_dotenv()

client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]},
)

messages = [
    {
        "role": "user",
        "content": "Why is the sky blue?",
    },
]

for part in client.chat("qwen3.5:397b-cloud", messages=messages, stream=True):
    print(part["message"]["content"], end="", flush=True)
