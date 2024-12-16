import json
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core import get_response_synthesizer, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.base.llms.types import ChatMessage, MessageRole


def query_rag(index, llm, query):
    
    vector_store = PineconeVectorStore(pinecone_index=index)
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    response_synthesizer = get_response_synthesizer()
    
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    #TODO: finalise use of query_engine or use of context to query LLM.
    llm_query = query_engine.query(query)
    details = {}
    for i in range(2):
        key = "context " + str(i)
        details[key] = llm_query.source_nodes[i].text
        i += 1
    retrieved_json = json.dumps(details, indent=2)
    
    prompt = f"""Based on the following retrieved data in json format: {retrieved_json}, please provide a detailed answer to the following query: {query}"""
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n 1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content= prompt,
        ),
    ]
    response = llm.chat(messages=messages)
    return str(response)[10:]