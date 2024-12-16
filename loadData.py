from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document

import cleanData

class DataLoader: 
    def __init__(self):
        pass

    def split_text_with_overlap(self, text, max_chunk_size=512, overlap=50):
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + max_chunk_size, len(text))
                chunks.append(text[start:end])
                start += (max_chunk_size - overlap)
            return chunks

    def load_data(self, filename="hnsw.pdf"):
        print(filename)
        inputfile =[]
        inputfile.append(filename)
        reader = SimpleDirectoryReader(input_dir="uploaded_files")
        documents = reader.load_data() 

        #TODO:
            #Implement for creating chucks based on context: SemanticSplitterNodeParser, SentenceSplitter
        
        cleaned_docs = []
        for d in documents:
            cleaned_text = cleanData.clean_up_text(d.text)
            d.text = cleaned_text
            cleaned_docs.append(d)

        #TODO: Add metadata depending on uploaded documents

        processed_documents = []
        for doc in cleaned_docs:
            chunks = self.split_text_with_overlap(doc.text, max_chunk_size=512, overlap=50)
            for chunk in chunks:
                processed_documents.append({
                    "text": chunk,
                    "metadata": doc.metadata  # Retain metadata for each chunk
                })

        # Convert processed documents back to Document objects
        final_documents = [Document(text=doc["text"], metadata=doc["metadata"]) for doc in processed_documents]
        # return cleaned_docs
        return final_documents



