from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not configured.")

if not model or model == "your_selected_model":
    raise RuntimeError("OPENAI_MODEL is not configured.")

client = OpenAI(api_key=api_key)

