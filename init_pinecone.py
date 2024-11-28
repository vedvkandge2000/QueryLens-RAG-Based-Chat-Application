import config
from pinecone import Pinecone, ServerlessSpec
import time
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini


pinecone_api_key = config.get_pinecone_api_key()

def initialize_llm(temperature, max_tokens):
    llm = Gemini(
    model="models/gemini-1.5-flash", api_key=config.get_gemini_api_key(), max_tokens = max_tokens, temperature = temperature)
    return llm

def initialize_pinecone():
    pc = Pinecone(api_key=pinecone_api_key)
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
    return index;

def reset_pinecone_index():
    index = initialize_pinecone()
    index.delete(delete_all=True)

def get_embed_model():
    #TODO: Try better embedding model for better accuracy
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return embed_model

def create_indexes(documents, index, embed_model):
  
    # Initialize VectorStore
    vector_store = PineconeVectorStore(pinecone_index=index)
   
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # TODO: Update settings if required
        # Settings.chunk_size = 256
        # Settings.chunk_overlap = 20
    Settings.embed_model = embed_model
    
    # Create index with the new settings
    Vindex = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    return
