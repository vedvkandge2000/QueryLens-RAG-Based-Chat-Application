from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings

def query_rag(index, llm, embed_model, query):
    Settings.llm = llm
    Settings.embed_model = embed_model
    vector_store = PineconeVectorStore(pinecone_index=index)
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Grab 5 search results
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    llm_query = query_engine.query(query)
    context = "Context:\n"
    for i in range(5):
        context = context + llm_query.source_nodes[i].text + "\n"

    # print(context)
    #TODO: Using context, query LLM model to get generated response based on context.
    res = llm_query.response
    return res