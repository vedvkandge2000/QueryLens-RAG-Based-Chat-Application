import streamlit as st
import os
from main import RAGChatbot
from loadData import DataLoader

# Directory to store uploaded files
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize the chatbot
chatbot = RAGChatbot()
data_loader = DataLoader()

# Streamlit UI
st.title("RAG-based Chatbot with File Upload")
st.subheader("Upload a file to create a knowledge base and interact with the chatbot!")

# Sidebar for configuration
st.sidebar.title("Settings")
temperature = st.sidebar.slider("Response Temperature", 0.0, 1.0, 0.7, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 100, step=10)
st.sidebar.write("Adjust the model's behavior to suit your needs.")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded file to the local directory
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded and saved as: {file_path}")

    # Process the file
    try:
        st.info("Processing the uploaded file...")
        
        # Pass the file path to the backend
        documents = data_loader.load_data(file_path)

        # Index the cleaned documents into Pinecone
        chatbot.index_documents(documents)
        st.success("File processed and indexed successfully!")
    except Exception as e:
        st.error(f"Error processing the file: {e}")

# Chat interface
user_query = st.text_input("Your Query:", placeholder="Type your question here...")

if st.button("Ask"):
    if user_query:
        with st.spinner("Fetching answer..."):
            response = chatbot.get_response(user_query, temperature=temperature, max_tokens=max_tokens)
        st.success("Here's the response:")
        st.write(response)
    else:
        st.warning("Please type a query to get started.")

# Display retrieved documents if needed
if st.checkbox("Show Retrieved Documents"):
    docs = chatbot.get_retrieved_docs(user_query)
    st.subheader("Retrieved Documents:")
    for doc in docs:
        st.write(doc)