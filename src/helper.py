from typing_extensions import TypedDict, List, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from dotenv import load_dotenv
from src.state import AgentState, GradeAnswer, GradeDocuments, GradeHallucinations
from prompt.state_prompt import *
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

DATA_PATH = "Data/attention-is-all-you-need-Paper.pdf"
DB_FAISS_PATH = 'faiss_index'

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

def create_model(state: AgentState) -> AgentState:
    state['model'] = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
    return state

def get_relevent_document(state: AgentState) -> AgentState:
    question = state["question"]
    vector_store = state["vector_store"]
    
    documents = vector_store.similarity_search(question)
    state["documents"] = documents
    return state

def grade_document(state: AgentState) -> AgentState:
    question = state["question"]
    documents = state["documents"]
    
    str_llm_output = state["model"].with_structured_output(GradeDocuments)
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DOCUMENT_GRADER_PROMPT),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    retriver_grader = grade_prompt | str_llm_output
    
    filter_documents = []
    for d in documents:
        score = retriver_grader.invoke({
            "question": question,
            "document": d.page_content
        })
        grade = score.binary_score
        if grade == "yes":
            filter_documents.append(d)
    
    state["documents"] = filter_documents
    return state    

def decide_to_generate(state: AgentState) -> str:
    filter_documents = state["documents"] 
    
    if not filter_documents:
        return "end"
    
    else:
        return "continue"

def generate_answer(state: AgentState) -> AgentState:
    question = state['question']
    documents = state["documents"]
    prompt = hub.pull("rlm/rag-prompt")
    
    chain = prompt | state["model"] | StrOutputParser()
    
    generate = chain.invoke({"context": documents, "question": question})
    
    state["generation"] = generate
    
    return state

def check_hallucination(state: AgentState) -> AgentState:
    documents = state['documents']
    generation = state['generation']
    
    str_llm_output = state['model'].with_structured_output(GradeHallucinations)
    hall_prompt = ChatPromptTemplate.from_messages([
        ("system", HALLUCINATION_GRADER_PROMPT),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    
    retriver_chain = hall_prompt | str_llm_output
    
    score = retriver_chain.invoke({"documents": documents, "generation": generation})
    
    grade = score.binary_score
    
    if grade == "yes":
        state["hallucination"] = False
    else:
        state["hallucination"] = True
        
    return state        

def grade_answer(state: AgentState) -> AgentState:
    question = state["question"]
    generation = state["generation"]
    
    str_llm_output = state['model'].with_structured_output(GradeAnswer)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_GRADER_PROMPT),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]) 
    
    chain = answer_prompt | str_llm_output
    score = chain.invoke({"question": question, "generation": generation})
    
    grade = score.binary_score
    if grade == "yes":
        state["valid_answer"] = True
    else:
        state["valid_answer"] = False
    return state
