import json, chromadb

with open ('papers.json', encoding='utf-8') as f:
    papers = json.load(f)


# Set-up chroma vector db
client = chromadb.PersistentClient('./chroma_db')
collection = client.get_or_create_collection('rag_papers')

collection.add(
    ids = [p["id"] for p in papers],
    documents = [f"{p['title']}\n\n{p['abstract']}" for p in papers],
    metadatas=[{"title": p['title'], "url": p["url"]} for p in papers]
)

print(f"✅ Indexed {collection.count()} papers into Chroma")