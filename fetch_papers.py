import arxiv, json

search = arxiv.Search(
    query='abs:"retrieval augmented generation" OR abs:"retrieval-augmented generation"',
    max_results=25,
    sort_by=arxiv.SortCriterion.Relevance,
)

papers = []
for r in arxiv.Client().results(search):
    papers.append({
        "id": r.entry_id.split("/")[-1],   # arxiv id → for citations later
        "title": r.title,
        "abstract": r.summary,
        "authors": [a.name for a in r.authors],
        "url": r.pdf_url,
    })

with open("papers.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"Saved {len(papers)} papers to papers.json")
