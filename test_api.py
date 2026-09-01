import os

from dotenv import load_dotenv
from openai import OpenAI


# Load variables from .env
load_dotenv()


# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")


if not api_key:
    print("OPENAI_API_KEY not found in .env")
    exit()


# Create OpenAI client
client = OpenAI(
    api_key=api_key
)


# Test API
try:

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Say hello in one sentence."
            }
        ]
    )

    print("\n--- OPENAI RESPONSE ---")
    print(response.choices[0].message.content)

except Exception as e:

    print("\nOpenAI API Error:")
    print(e)