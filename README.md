# Agentic RAG System

This repository contains an implementation of an Agentic Retrieval-Augmented Generation (RAG) system using LangChain and LangGraph. The system enhances traditional RAG by adding decision-making capabilities to improve query understanding and document relevance.

## Architecture

![Agentic RAG Workflow](rag.png)

The workflow follows these steps:

1. **AI Agent**: Processes the user query and decides whether to retrieve information
2. **Retriever**: Fetches relevant documents from the vector store
3. **Document Grader**: Evaluates if retrieved documents are relevant to the query
4. **Generator**: Creates a response based on retrieved documents
5. **Query Rewriter**: Reformulates queries that didn't yield relevant documents

## Features

- Uses Groq's Gemma2-9b-It model for AI reasoning
- HuggingFace embeddings (All-MiniLM-L6-v2) for document vectorization
- Document chunking with RecursiveCharacterTextSplitter
- Vector storage with Chroma
- Relevance evaluation with a binary grading system
- Query reformulation for improved search results

## Requirements

- Python 3.10 or higher
- Dependencies listed in requirements.txt:
  - langgraph
  - langchain
  - langchain_community
  - langchain_huggingface
  - langchainhub
  - langchain_groq
  - ipykernel
  - dotenv
  - bs4
  - tiktoken
  - chromadb
  - IPython

## Setup

1. Clone this repository
```bash
git clone https://github.com/Zoro-chi/agentic-rag.git
cd agentic-rag
```

2. Create and activate a conda environment
```bash
conda create -n agentic-rag python=3.10
conda activate agentic-rag
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys
```
GROQ_API_KEY=your_groq_api_key
```

5. Run the Jupyter notebook
```bash
jupyter notebook agentic_rag.ipynb
```

## Usage

The system can be used by invoking the compiled workflow:

```python
# Define your question
messages = [HumanMessage(content="What is the difference between RAG and LLMs?")]

# Invoke the app
app.invoke({"messages": messages})
```

## How It Works

1. The AI agent processes the user query and decides whether to use the retrieval tool
2. The retriever searches for relevant documents in the vector store
3. The document grader determines if the retrieved documents are relevant:
   - If relevant, the query and documents are passed to the generator
   - If not relevant, the query is sent to the rewriter to reformulate it
4. The system returns a comprehensive response based on the retrieved information

## License

MIT