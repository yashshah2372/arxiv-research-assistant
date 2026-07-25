import os, chromadb
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",   # ← only line that's Groq-specific
)
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("rag_papers")

def answer(question: str, k: int = 3):
    # 1. Retrieve top-k papers
    results = collection.query(query_texts=[question], n_results=k)
    docs, metas = results["documents"][0], results["metadatas"][0]

    # 2. Build numbered context (the citation source)
    context = "".join(
        f"[{i}] {m['title']}\n{d}\n\n"
        for i, (d, m) in enumerate(zip(docs, metas), 1)
    )

    # 3. Ground the LLM: answer only from context, cite with [n], else say "I don't know"
    prompt = f"""Answer the question using ONLY the context below.
Cite the sources you use with their [number]. If the context does not
contain the answer, say you don't have enough information.

Context:
{context}
Question: {question}"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return resp.choices[0].message.content, metas

if __name__ == "__main__":
    ans, sources = answer("How do you evaluate a RAG system?")
    print(ans)
    print("\nSources:")
    for i, m in enumerate(sources, 1):
        print(f"[{i}] {m['title']}")
