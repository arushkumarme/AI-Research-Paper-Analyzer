from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


MODEL_NAME = "gemini-2.0-flash"


def generate_answer(query, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a professional research AI assistant.

Answer ONLY using provided context.
If answer is missing, say you cannot find it.

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text
