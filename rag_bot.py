"""
Orient BlackSwan History Bot — RAG QA System
=============================================
An end-to-end Retrieval-Augmented Generation chatbot that answers
user questions *strictly* from the Orient BlackSwan History Catalogue
2024-25 (``History_2024-25.pdf``).

Usage
-----
    # Interactive mode (default)
    python rag_bot.py

    # Batch-evaluate the 5 project questions
    python rag_bot.py --batch

    # Re-build the vector store from scratch
    python rag_bot.py --rebuild

Environment
-----------
    OPENAI_API_KEY   – Required for the LLM (GPT-4o / GPT-3.5-turbo).
    EMBEDDING_TYPE   – "huggingface" (default, free) or "openai".
    LLM_MODEL        – "gpt-4o" (default) or "gpt-3.5-turbo".
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# ── LangChain core ──────────────────────────────────────────────
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Document loaders ────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader

# ── Embeddings ──────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# ── Vector store ────────────────────────────────────────────────
from langchain_chroma import Chroma

# ── LLM ─────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
load_dotenv()  # reads .env in cwd

PDF_PATH: str = os.getenv("PDF_PATH", "data/History_2024-25.pdf")
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_TYPE: str = os.getenv("EMBEDDING_TYPE", "huggingface")  # "huggingface" | "openai"
HF_MODEL_NAME: str = os.getenv(
    "HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K: int = int(os.getenv("TOP_K", "5"))

console = Console()

# ═══════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=textwrap.dedent("""\
        You are the **Orient BlackSwan History Bot**, an expert assistant
        that answers questions ONLY from the Orient BlackSwan History
        Catalogue 2024-25.

        ── RULES ──────────────────────────────────────────────────
        1. Answer using ONLY the information present in the CONTEXT below.
        2. For every fact you state, provide the exact reference in the
           format:  **[Page <N> | <Book/Section title>]**
        3. If the CONTEXT does not contain enough information to answer
           the question, reply EXACTLY with:

               Not found — the requested information is not present in
               the Orient BlackSwan History Catalogue 2024-25.

           Do NOT speculate or use external knowledge.
        4. If the question is subjective (e.g. "best" book), state that
           it is subjective and either list relevant options from the
           context or reply "Not found" if none are relevant.
        ──────────────────────────────────────────────────────────

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
    """),
)

# ═══════════════════════════════════════════════════════════════
# 5 PROJECT QUESTIONS  (for batch evaluation)
# ═══════════════════════════════════════════════════════════════
PROJECT_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "Q1",
        "question": "On what date were the prices in this catalogue last updated?",
        "expected": (
            "The prices were last updated on 20-01-2025. "
            "[Reference: Catalogue cover / introductory page]"
        ),
    },
    {
        "id": "Q2",
        "question": "In which year was the publisher (Orient BlackSwan) incorporated?",
        "expected": (
            "Orient BlackSwan was incorporated in the year 1948. "
            "[Reference: About the publisher / inside cover]"
        ),
    },
    {
        "id": "Q3",
        "question": "What is the price of a Physics textbook?",
        "expected": (
            "Not found — this is a History catalogue and does not list "
            "Physics textbooks."
        ),
    },
    {
        "id": "Q4",
        "question": "Who is the current Minister of Education?",
        "expected": (
            "Not found — the catalogue does not contain information about "
            "the current Minister of Education."
        ),
    },
    {
        "id": "Q5",
        "question": 'Which is the "best" history book for a beginner?',
        "expected": (
            "Not found / Subjective — the catalogue does not designate any "
            'single book as the "best" for beginners. The bot may list '
            "relevant introductory titles if present in context."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════


def _get_embeddings():
    """Return an embedding model based on ``EMBEDDING_TYPE``."""
    if EMBEDDING_TYPE.lower() == "openai":
        console.print("[cyan]Using OpenAI embeddings[/]")
        return OpenAIEmbeddings()
    console.print(f"[cyan]Using HuggingFace embeddings: {HF_MODEL_NAME}[/]")
    return HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_split_pdf(pdf_path: str) -> list:
    """Load a PDF and split it into overlapping chunks."""
    path = Path(pdf_path)
    if not path.exists():
        console.print(
            f"[bold red]ERROR:[/] PDF not found at [yellow]{path.resolve()}[/].\n"
            "Place the file in the project root or set PDF_PATH in .env."
        )
        sys.exit(1)

    console.print(f"[green]Loading PDF:[/] {path.resolve()}")
    loader = PyPDFLoader(str(path))
    pages = loader.load()                       # each element = one page
    console.print(f"  → {len(pages)} page(s) loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    console.print(f"  → {len(chunks)} chunks created (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def build_vectorstore(chunks: list, persist_dir: str) -> Chroma:
    """Create (or overwrite) a Chroma vector store on disk."""
    embeddings = _get_embeddings()
    console.print("[green]Building Chroma vector store…[/]")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="history_catalogue",
    )
    console.print(f"  → Stored at [yellow]{Path(persist_dir).resolve()}[/]")
    return vectorstore


def load_vectorstore(persist_dir: str) -> Optional[Chroma]:
    """Load an existing Chroma vector store; return None if absent."""
    p = Path(persist_dir)
    if not p.exists() or not any(p.iterdir()):
        return None
    embeddings = _get_embeddings()
    console.print("[green]Loading existing vector store…[/]")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="history_catalogue",
    )


def get_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """Wrap the retriever + LLM into a RetrievalQA chain."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,          # deterministic for evaluation
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain


# ═══════════════════════════════════════════════════════════════
# QUERY EXECUTION
# ═══════════════════════════════════════════════════════════════


def ask(chain: RetrievalQA, question: str) -> dict:
    """Run a single query through the chain and return structured output."""
    result = chain.invoke({"query": question})
    answer: str = result.get("result", "")
    sources = result.get("source_documents", [])

    # Deduplicate pages used
    page_refs: list[str] = sorted(
        {str(doc.metadata.get("page", "?") + 1) for doc in sources}   # 0-indexed → 1-indexed
    )
    return {
        "question": question,
        "answer": answer.strip(),
        "pages_referenced": page_refs,
        "source_chunks": sources,
    }


def display_answer(res: dict) -> None:
    """Pretty-print a single answer."""
    console.print(Panel(Markdown(res["answer"]), title="📖 Answer", border_style="green"))
    if res["pages_referenced"]:
        console.print(
            f"  📄 Source pages: {', '.join(res['pages_referenced'])}\n"
        )


# ═══════════════════════════════════════════════════════════════
# BATCH EVALUATION
# ═══════════════════════════════════════════════════════════════


def run_batch(chain: RetrievalQA) -> None:
    """Evaluate the 5 project questions and print a comparison table."""
    table = Table(
        title="📋 Batch Evaluation — 5 Project Questions",
        show_lines=True,
        width=120,
    )
    table.add_column("ID", style="bold cyan", width=4)
    table.add_column("Question", width=35)
    table.add_column("Bot Answer", width=38)
    table.add_column("Expected", width=35)

    for q in PROJECT_QUESTIONS:
        console.print(f"\n[bold yellow]▶ {q['id']}:[/] {q['question']}")
        res = ask(chain, q["question"])
        display_answer(res)

        # Truncate for table readability
        bot_short = (res["answer"][:120] + "…") if len(res["answer"]) > 120 else res["answer"]
        exp_short = (q["expected"][:120] + "…") if len(q["expected"]) > 120 else q["expected"]

        table.add_row(q["id"], q["question"], bot_short, exp_short)

    console.print("\n")
    console.print(table)


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════


def interactive(chain: RetrievalQA) -> None:
    """Run a REPL-style interactive Q&A loop."""
    console.print(
        Panel(
            "[bold green]Orient BlackSwan History Bot[/]\n"
            "Ask any question about the History Catalogue 2024-25.\n"
            'Type [bold yellow]"quit"[/] or [bold yellow]"exit"[/] to stop.',
            border_style="bright_blue",
        )
    )
    while True:
        try:
            question = console.input("[bold cyan]❓ You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Goodbye! 👋[/]")
            break
        res = ask(chain, question)
        display_answer(res)


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orient BlackSwan History Bot — RAG QA System",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run the 5 project questions in batch mode.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force-rebuild the Chroma vector store from the PDF.",
    )
    args = parser.parse_args()

    # ── Step 1: Vector store ────────────────────────────────────
    vectorstore = None
    if not args.rebuild:
        vectorstore = load_vectorstore(CHROMA_DIR)

    if vectorstore is None:
        chunks = load_and_split_pdf(PDF_PATH)
        vectorstore = build_vectorstore(chunks, CHROMA_DIR)

    # ── Step 2: Build QA chain ──────────────────────────────────
    chain = get_qa_chain(vectorstore)

    # ── Step 3: Run mode ────────────────────────────────────────
    if args.batch:
        run_batch(chain)
    else:
        interactive(chain)


if __name__ == "__main__":
    main()
