import init_pinecone
import rag

class RAGChatbot:
    def __init__(self):
        init_pinecone.get_embed_model()
        self.pc = init_pinecone.get_pinecone_instance()

    def get_response(self, query, temperature, max_tokens):
        """Generate a response to the user's query."""
        try:
            response = rag.query_rag(init_pinecone.initialize_pinecone(self.pc), init_pinecone.initialize_llm(temperature, max_tokens),  query=query)
            
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_retrieved_docs(self, query):
        """Retrieve relevant documents for the query."""
        try:
            docs = self.rag_model.retrieve_documents(query)
            return docs
        except Exception as e:
            return [f"Error retrieving documents: {str(e)}"]
        
    def index_documents(self, documents):
        try:
            init_pinecone.reset_pinecone_index(self.pc)
            init_pinecone.create_indexes(documents, init_pinecone.initialize_pinecone(self.pc))
        except Exception as e:
            return [f"Error indexing documents: {str(e)}"]

if __name__ == "__main__":
    chatbot = RAGChatbot()
    print("Chatbot initialized. Ready to answer queries!")

