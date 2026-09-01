from dotenv import dotenv_values
from google import genai


# Load .env
config = dotenv_values(".env")

api_key = config.get("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found!")
    exit()


print("Gemini API key loaded successfully! ✅")


# Create Gemini client
client = genai.Client(
    api_key=api_key
)


# Send request using the current Interactions API
interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Explain RAG in one simple sentence."
)


print("\n--- GEMINI RESPONSE ---")
print(interaction.output_text)