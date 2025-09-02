from src.helper import *
from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from IPython.display import Image

class BuildGraph:
    def __init__(self) -> None:
        pass
    
    def build_graph(self):
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("create_model", create_model)
        workflow.add_node("get_relevant_documents", get_relevent_document)
        workflow.add_node("grade_documents", grade_document)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("check_for_hallucination", check_hallucination)
        workflow.add_node("grade_answer", grade_answer)

        # Build graph
        workflow.add_edge(START, "create_model")
        workflow.add_edge("create_model", "get_relevant_documents")
        workflow.add_edge("get_relevant_documents", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "continue": "generate_answer",
                "end": END,
            },
        )
        workflow.add_edge("generate_answer", "check_for_hallucination")
        workflow.add_edge("check_for_hallucination", "grade_answer")

        # Compile
        return workflow.compile()
    
    def display_graph(self):
        graph = self.build_graph()
        return Image(graph.get_graph().draw_mermaid_png())
