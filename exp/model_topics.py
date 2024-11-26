import newsrag.feeds as feeds
from pathlib import Path
import json
import arrow

from haystack.document_stores.in_memory import InMemoryDocumentStore

import newsrag.pipelines as pipelines
import yaml
import jsonlines
from sklearn.metrics import silhouette_score

import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r"))["model_topics"]


    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    doc_store_file = data_dir / "document_store.json"
    doc_store = InMemoryDocumentStore.load_from_disk(doc_store_file)

    topic_model = pipelines.TopicModelPipeline(doc_store, umap_args=params["umap"], hdbscan_args=params["hdbscan"])
    topic_results = topic_model.run(arrow.utcnow().shift(days=-params["days"]))
    model_results = topic_results["topic_model"]

    documents = model_results["documents"]
    embeddings = [d.embedding for d in documents]
    topic_ids = [d.meta["topic_id"] for d in documents]
    silhouette = silhouette_score(embeddings, labels=topic_ids)

    # serialize to json-lines
    # out_file =  data_dir / "topic_documents.jsonl"
    # with jsonlines.Writer(out_file.open("w"), dumps=NpEncoder().default) as writer:
    #     for doc in model_results["documents"]:
    #         writer.write(doc.to_dict())

    metrics_file = data_dir / "model_topics.json"
    with metrics_file.open("w") as fh:
        json.dump({
            "total_topics": len(model_results["topic_words"]),
            "silhouette_score": silhouette
        }, fh)