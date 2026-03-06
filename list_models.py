from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

models = client.models.list()

print("사용 가능한 Groq 모델 목록:")
for model in models.data:
    print(f"- {model.id}")
