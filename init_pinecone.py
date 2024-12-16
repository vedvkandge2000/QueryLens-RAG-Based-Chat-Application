import config
from pinecone import Pinecone, ServerlessSpec
import time
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.lmstudio import LMStudio

pinecone_api_key = config.get_pinecone_api_key()

def initialize_llm(temperature, max_tokens):
    llm = LMStudio(
        model_name="mathstral-7b-v0.1",
        base_url="http://localhost:1234/v1",
        temperature=temperature,
        num_output=max_tokens
    )
    Settings.llm = llm
    return llm

def get_pinecone_instance():
    pc = Pinecone(api_key=pinecone_api_key)
    return pc;

def initialize_pinecone(pc):
    
    index_name = "llama-integration-pinecone-v1"
    
    if not pc.has_index(index_name):
        pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(5)
    index = pc.Index(index_name)
    return index

def reset_pinecone_index(pc):
    index_name = "llama-integration-pinecone-v1"
    if pc.has_index(index_name):
        index = pc.Index(index_name)
        index.delete(delete_all=True)
        print("Index resetted successfully")

def get_embed_model():
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.embed_model = embed_model

def create_indexes(documents, index):
  
    vector_store = PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index with the new settings
    Vindex = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    return
