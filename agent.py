import os, chromadb
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("rag_papers")


TOP_K = 4


@tool
def search_papers(query: str) -> str:
    """Search the arXiv corpus of RAG papers and return the closest matches as
    numbered entries with title, URL and abstract. Search one topic at a time:
    for a question covering several topics, call this once per topic."""
    results = collection.query(query_texts=[query], n_results=TOP_K)
    docs, metas = results["documents"][0], results["metadatas"][0]
    if not docs:
        return "No matching papers found."
    return "\n\n".join(
        f"[{i}] {m['title']}\nURL: {m['url']}\n{d}"
        for i, (d, m) in enumerate(zip(docs, metas), 1)
    )


SYSTEM = """You are a research assistant over a corpus of arXiv papers on RAG.

Gather evidence with the search_papers tool before you answer. If a question
spans several topics, search each one separately instead of in a single broad
query. Answer using ONLY what the tool returned, citing papers by title. If the
results don't cover the question, say you don't have enough information."""

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

# The LLM decides *whether* and *how* to call search_papers; LangGraph runs the
# Thought → Action → Observation loop and feeds each result back to the model.
agent = create_react_agent(llm, tools=[search_papers], prompt=SYSTEM)


if __name__ == "__main__":
    question = (
        "How do papers evaluate RAG systems, and what do they say about "
        "reranking? Compare across papers."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    for m in result["messages"]:
        m.pretty_print()
