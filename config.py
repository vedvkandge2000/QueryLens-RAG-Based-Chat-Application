import os
from dotenv import load_dotenv
load_dotenv()

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')
hf_token = os.environ.get('HF_TOKEN')
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Add error handling for missing keys
if not pinecone_api_key or not openai_api_key or gemini_api_key or hf_token:
    raise ValueError("Missing required API keys. Please ensure PINECONE_API_KEY and GEMINI_API_KEY are set in your environment.")

def get_pinecone_api_key():
    return pinecone_api_key

def get_openai_api_key():
    return openai_api_key   

def get_gemini_api_key():
    return gemini_api_key