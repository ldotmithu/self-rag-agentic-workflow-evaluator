from src.workflow import BuildGraph
from src.state import AgentState
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
import os

# For colored terminal text
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.table import Table
except ImportError:
    raise ImportError("Please install 'rich' for better console UI: pip install rich")

warnings.filterwarnings("ignore")
console = Console()


def main():
    console.print(Panel.fit("🤖 [bold cyan]Self-RAG Chat Assistant[/bold cyan] 🤖", border_style="cyan"))
    console.print("[green]Type 'quit' anytime to exit[/green]\n")

    workflow = BuildGraph()
    graph = workflow.build_graph()

    # Create default model
    default_model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

    # Load the actual vector store
    DB_FAISS_PATH = 'faiss_index'
    embedding_model = HuggingFaceEmbeddings(
        model_kwargs={},
        encode_kwargs={'normalize_embeddings': True}
    )

    try:
        vector_store = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        console.print("[bold green]✅ Vector store loaded successfully.[/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]❌ Error loading vector store:[/bold red] {e}")
        console.print("[yellow]Run 'python src/build_index.py' to create the FAISS index first.[/yellow]")
        return

    # Chat loop
    while True:
        question = Prompt.ask("[bold blue]Ask[/bold blue]")
        if question.lower() == "quit":
            console.print("[bold red]👋 Exiting Self-RAG Assistant. Goodbye![/bold red]")
            break

        # Prepare initial state
        initial_state = AgentState(
            question=question,
            generation="",
            documents=[],
            model=default_model,
            vector_store=vector_store,
            hallucination=False,
            valid_answer=False
        )

        # Run workflow
        response = graph.invoke(initial_state)

        console.print("\n[bold magenta]--- QUALITY ASSESSMENT RESULTS ---[/bold magenta]")

        # Create assessment table
        assessment_table = Table(show_header=True, header_style="bold magenta")
        assessment_table.add_column("Metric", style="cyan")
        assessment_table.add_column("Status", style="bold")
        assessment_table.add_column("Details", style="white")

        # Hallucination status
        hallucination_status = response.get("hallucination", False)
        if hallucination_status:
            assessment_table.add_row("Hallucination", "[red]❌ DETECTED[/red]", "Answer may not be grounded in documents")
        else:
            assessment_table.add_row("Hallucination", "[green]✅ CLEAN[/green]", "Answer is grounded in facts")

        # Validity status
        validity_status = response.get("valid_answer", True)
        if not validity_status:
            assessment_table.add_row("Answer Validity", "[yellow]⚠️ QUESTIONABLE[/yellow]", "Answer may not fully address the question")
        else:
            assessment_table.add_row("Answer Validity", "[green]✅ VALID[/green]", "Answer properly addresses the question")

        console.print(assessment_table)
        console.print("\n")

        # Display the generated answer
        if "generation" in response:
            console.print(Panel.fit(response["generation"], border_style="green", title="💡 Generated Answer"))

        # Summary message
        if not hallucination_status and validity_status:
            console.print("\n[bold green]🎉 Excellent! The answer is both factual and relevant.[/bold green]")
        elif not hallucination_status and not validity_status:
            console.print("\n[bold yellow]⚠️ The answer is factual but may not fully address your question.[/bold yellow]")
        elif hallucination_status and validity_status:
            console.print("\n[bold red]❌ Warning: The answer addresses the question but contains potential hallucinations.[/bold red]")
        else:
            console.print("\n[bold red]❌ Critical: The answer contains hallucinations and doesn't properly address the question.[/bold red]")

        console.print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
