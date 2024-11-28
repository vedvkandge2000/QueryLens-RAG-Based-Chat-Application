# QueryLens: RAG Based Chat Application
QueryLens is an AI-powered tool that revolutionizes how developers/configurators interact with and understand documentations.

## Inspiration

The inspiration for QueryLens, a RAG-based chat application for document interaction, came from observing the challenges faced by professionals across industries when working with extensive and complex documentation:

1. Configuration Complexity : Users often struggle to find precise answers to their configuration-related queries when navigating through lengthy manuals, guides, or technical documents.A chatbot tailored for documentation allows users to ask specific questions and receive instant, accurate answers without having to comb through pages of text.

2. Streamlining Workflows: Configuring components based on technical or setup documents often requires users to follow detailed instructions spread across multiple sections. This tool can create an interactive flow to help users configure components efficiently by guiding them step-by-step and providing contextual information directly from the document.

3. Time Efficiency: Manually searching for the right information within large documents can be tedious and time-consuming, especially when deadlines are tight.QueryLens significantly reduces time spent searching by offering on-demand, conversational access to document knowledge.

4. Critical Information Extraction: Users may overlook important configurations or miss essential details when trying to digest dense documentation.QueryLens ensures critical configurations and key insights are easily surfaced and highlighted during interactions, improving accuracy and productivity.

5. Personalized Assistance: Users can interact conversationally with the document, asking tailored questions and receiving responses specific to their needs. This makes the documentation more accessible and approachable, even for users with less technical expertise.



## 🚀 Features

+ 💬 AI-Driven Chatbot
  - Ask natural language questions about your documents
  - Receive instant, contextually accurate answers

+ 📊 Commit Summarization
  - Create intelligent, concise summaries to configure any flow or component from your document
  - Quickly grasp changes with ease

## 🛠️ Built With

  - Frontend: [Streamlit.io](https://streamlit.io)
  - Backend: Python, [Llamaindex.ai](https://www.llamaindex.ai)
  - AI/ML:
    - [Pinecone](https://www.pinecone.io/) for vector storage
    - [Gemini](https://gemini.google.com)
    - [HuggingFace](https://huggingface.co) for embedding model
  - Version control: Git (via [pythonGit](https://gitpython.readthedocs.io/))
    
## 🏁 Getting Started

  ### Prerequisites

  - Python
  - Llamaindex
  - Pinecone
  - Streamlit
  - Git
  - Gemini

  ### Installation

  1. Clone the repo
  ```
https://github.com/vedvkandge2000/QueryLens-RAG-Based-Chat-Application.git
  ```
  2. Install Python dependencies
  ```
  pip install -r requirements.txt
  ```
  3. Create API keys and update .env file
     - Create Pinecone API
     - Create Gemini API
     - Create HuggingFace Token
       
  4. Run the application
  ```
  streamlit run app.py
  ```
  Now QueryLens in action! 🎮 Usage Chatbot Interface

  * Navigate to the Chatbot, Upload your document and wait for application to process it. After uploading document click clear symbol to clear the document.
  * Now your chatbot is ready, to answer any question regarding your document. Type your question about the document (e.g., "What is this text about?") Hit enter and wait for the AI-generated response.

## TODO
  - Create chucks while indexing based on context.
  - Add chuck index specific metadata for each chuck.
  - Integrate LLM model to use generated response and context, to create a response.
  - Try better embedding models and LLM models.

