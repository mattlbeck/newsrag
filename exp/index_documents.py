import json
from pathlib import Path

import jsonlines
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

import newsrag.pipelines as pipelines
import newsrag.topics as topics

if __name__ == "__main__":
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # load documents from previous stage
    docs_file = data_dir / "documents.jsonl"
    docs = []
    with jsonlines.open(docs_file) as reader:
        for doc in reader:
            docs.append(Document.from_dict(doc))

    # index documents
    store = InMemoryDocumentStore()
    
    embedder = topics.SentenceTransformersJointEmbedder()
    indexing = pipelines.JointDocumentIndexingPipeline(store, joint_embedder=embedder)
    indexing.run(docs)

    words = store.filter_documents({"field": "type", "operator": "==", "value": "word"})
    documents = store.filter_documents({"field": "type", "operator": "==", "value": "document"})
    metrics_file = data_dir / "index_documents.json"
    with metrics_file.open("w") as fh:
        json.dump({
            "vocabulary_size": len(words),
            "total_documents": len(documents)
        }, fh)


    # serialize to the document store
    out_file =  data_dir / "document_store.json"
    store.save_to_disk(out_file)
