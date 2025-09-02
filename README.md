# 🤖 Self-RAG Agentic Workflow Evaluator

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange.svg)](https://www.langchain.com/langgraph)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A **Retrieval-Augmented Generation (RAG)** framework with self-reflection capabilities, powered by **LangChain**, **LangGraph**, and **Groq**. It evaluates document relevance, detects hallucinations, and assesses answer quality to build smarter, reliable RAG systems.

## ✨ Features

- 📚 **Semantic Document Retrieval**: Uses FAISS for efficient vector search.
- 🎯 **Relevance Grading**: LLM-powered scoring of retrieved documents.
- 🚨 **Hallucination Detection**: Verifies response factuality against documents.
- ✅ **Answer Quality Assessment**: Ensures responses are accurate and relevant.
- 🔄 **Agentic Workflow**: State-driven orchestration with LangGraph.
- 💬 **Interactive CLI**: Test the system via a command-line chat interface.

## 🏗️ Architecture

The system follows a multi-step workflow:

1. 📂 **Vector Store Creation**: Processes PDFs and builds a FAISS index.
2. 🔍 **Document Retrieval**: Retrieves documents based on semantic similarity.
3. 🎯 **Relevance Grading**: Evaluates document usefulness for the query.
4. ✍️ **Answer Generation**: Creates context-aware responses.
5. 🚨 **Hallucination Checking**: Validates response accuracy.
6. ✅ **Answer Grading**: Assesses response quality and relevance.

## 📂 Project Structure

```
self-rag-agentic-workflow-evaluator/
├── src/
│   ├── __init__.py
│   ├── build_index.py         # Builds FAISS vector store
│   ├── helper.py              # Core RAG functions
│   ├── state.py               # Workflow state and grading models
│   └── workflow.py            # LangGraph workflow definition
├── prompt/
│   ├── __init__.py
│   └── state_prompt.py        # Prompt templates
├── Data/
│   └── attention-is-all-you-need-Paper.pdf  # Sample document
├── local_chat.py              # CLI-based chat interface
├── notebook.ipynb             # Jupyter notebook for experimentation
└── README.md
```

## ⚡ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ldotmithu/self-rag-agentic-workflow-evaluator.git
   cd self-rag-agentic-workflow-evaluator
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the root directory:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 🚀 Usage

1. **Build the Vector Store**:
   ```bash
   python src/build_index.py
   ```

2. **Run the Interactive Chat**:
   ```bash
   python local_chat.py
   ```

3. **Use in Python Code**:
   ```python
   from src.workflow import BuildGraph

   workflow = BuildGraph()
   graph = workflow.build_graph()
   response = graph.invoke({"question": "What is Machine Learning?"})
   print(response)
   ```

## 🧩 Key Components

### 📌 State Management (`src/state.py`)
- **AgentState**: Manages workflow state.
- **GradeDocuments**: Scores document relevance.
- **GradeHallucinations**: Detects factual inaccuracies.
- **GradeAnswer**: Validates answer quality.

### 📌 Core Functions (`src/helper.py`)
- `create_model()`: Initializes the ChatGroq model.
- `build_vector_store()`: Creates the FAISS index.
- `get_relevant_documents()`: Retrieves relevant documents.
- `grade_document()`: Scores document relevance.
- `generate_answer()`: Produces context-aware responses.
- `check_hallucination()`: Identifies factual errors.
- `grade_answer()`: Evaluates response quality.

### 📌 Workflow (`src/workflow.py`)
- **BuildGraph**: Constructs the LangGraph workflow with conditional routing.
- Supports graph visualization.

## ⚙️ Configuration

- **Environment Variables**:
  - `GROQ_API_KEY`: API key for Groq LLM services.
- **Vector Store**:
  - Location: `faiss_index/`
  - Embeddings: HuggingFace Embeddings
  - Source: `Data/Final_Research_24474.pdf`

## 🧪 Development & Testing

```bash
# Test vector store creation
python src/build_index.py

# Test workflow compilation
python -c "from src.workflow import BuildGraph; g = BuildGraph().build_graph(); print('✅ Graph compiled')"

# Test chat interface
python local_chat.py
```

## 🤝 Contributing

1. 🍴 Fork the repository.
2. 🌱 Create a feature branch (`git checkout -b feature/YourFeature`).
3. 🛠️ Add improvements with clear comments.
4. ✅ Test thoroughly.
5. 🔄 Submit a pull request.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
- [Groq](https://groq.com/) for LLM services
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [HuggingFace](https://huggingface.co/) for embeddings