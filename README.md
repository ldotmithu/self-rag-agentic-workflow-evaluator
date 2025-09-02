# Self-RAG Agentic Workflow Evaluator

- Building a Self-RAG | Making your Agentic workflows critique through Self-Reflection | LangChain

- A Retrieval-Augmented Generation (RAG) system with self-evaluation capabilities that grades document relevance, checks for hallucinations, and assesses answer quality using LangGraph and LangChain.

## Features

- **Document Retrieval**: Vector-based semantic search using FAISS
- **Relevance Grading**: LLM-powered evaluation of document relevance
- **Hallucination Detection**: Automated checking for factual accuracy
- **Answer Quality Assessment**: Evaluation of generated answers
- **Agentic Workflow**: State-based graph workflow using LangGraph
- **Interactive Chat Interface**: Command-line interface for testing

## Architecture

The system follows a multi-step workflow:

1. **Vector Store Creation**: PDF document processing and embedding
2. **Document Retrieval**: Semantic search based on user queries
3. **Relevance Grading**: LLM evaluation of retrieved documents
4. **Answer Generation**: Context-aware response generation
5. **Hallucination Checking**: Factual accuracy verification
6. **Answer Grading**: Final quality assessment

## Project Structure

```
self-rag-agentic-workflow-evaluator/
├── src/
│   ├── __init__.py
│   ├── build_index.py          # Vector store creation
│   ├── helper.py              # Core RAG functions
│   ├── state.py               # Type definitions and models
│   └── workflow.py            # LangGraph workflow definition
├── prompt/
│   ├── __int__.py
│   └── state_prompt.py        # Prompt templates
├── Data/
│   └── attention-is-all-you-need-Paper.pdf"  # Example document
├── local_chat.py              # Interactive chat interface
├── loacl_chat.py              # Original chat interface (backup)
├── notebook.ipynb             # Jupyter notebook for experimentation
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ldotmithu/self-rag-agentic-workflow-evaluator.git
cd self-rag-agentic-workflow-evaluator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

## Usage

### 1. Create Vector Store
```bash
python src/build_index.py
```

### 2. Run Interactive Chat
```bash
python local_chat.py
```

### 3. Use in Code
```python
from src.workflow import BuildGraph

workflow = BuildGraph()
graph = workflow.build_graph()

# Invoke with a question
response = graph.invoke({
    "question": "Your question here"
})
```

## Key Components

### State Management (`src/state.py`)
- `AgentState`: TypedDict for workflow state management
- `GradeDocuments`: Pydantic model for document relevance scoring
- `GradeHallucinations`: Model for hallucination detection
- `GradeAnswer`: Model for answer quality assessment

### Core Functions (`src/helper.py`)
- `create_model`: Initialize ChatGroq model
- `get_relevent_document`: Retrieve relevant documents
- `grade_document`: Evaluate document relevance
- `generate_answer`: Generate responses
- `check_hallucination`: Detect factual inaccuracies
- `grade_answer`: Assess answer quality

### Workflow (`src/workflow.py`)
- `BuildGraph` class for creating the LangGraph workflow
- State-based conditional routing
- Visual graph generation capabilities

## Configuration

### Environment Variables
- `GROQ_API_KEY`: API key for Groq LLM service
- Model configuration in `src/helper.py`

### Vector Store
- Location: `faiss_index/` directory
- Embedding model: HuggingFace embeddings
- Document source: `Data/Final_Research_24474.pdf`

## API Reference

### Models
- **ChatGroq**: LLM provider for generation and evaluation
- **FAISS**: Vector store for document retrieval
- **HuggingFaceEmbeddings**: Text embedding model

### Workflow Nodes
1. `create_model` - Initialize LLM
2. `get_relevant_documents` - Retrieve documents
3. `grade_documents` - Evaluate relevance
4. `generate_answer` - Create response
5. `check_for_hallucination` - Verify facts
6. `grade_answer` - Assess quality

## Development

### Code Improvements Made
- ✅ Removed code duplication in helper functions
- ✅ Fixed function naming conventions
- ✅ Enhanced type safety with proper annotations
- ✅ Improved error handling throughout
- ✅ Optimized workflow structure
- ✅ Added comprehensive documentation

### Testing
```bash
# Test vector store creation
python src/build_index.py

# Test workflow compilation
python -c "from src.workflow import BuildGraph; graph = BuildGraph().build_graph(); print('Graph compiled successfully')"

# Test chat interface
python local_chat.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make improvements with proper type annotations
4. Test all changes thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with LangChain and LangGraph
- Uses Groq for LLM services
- FAISS for vector similarity search
- HuggingFace for embeddings

## Support

For issues and questions, please open an issue on the GitHub repository.
