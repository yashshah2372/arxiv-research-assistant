import chromadb

client = chromadb.PersistentClient(path = './chroma_db')

collection = client.get_collection('rag_papers')

results = collection.query(
    query_texts=['How do you evaluate a RAG system?'],
    n_results=3
)

for title in [m['title'] for m in results['metadatas'][0]]:
    print("*", title)