import newsrag.feeds as feeds
from pathlib import Path
import json
import arrow

from haystack.document_stores.in_memory import InMemoryDocumentStore

import newsrag.pipelines as pipelines
import yaml
import jsonlines
from sklearn.metrics import silhouette_score
from collections import defaultdict
import statistics
import numpy as np
from collections import Counter

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def evaluate_topics(model_results):
    documents = model_results["documents"]
    topic_ids = [d.meta["topic_id"] for d in documents]
    topic_sizes = Counter(topic_ids).values()
    silhouette = silhouette_score(model_results["umap_embedding"], labels=topic_ids)
    return {
            "total_topics": len(model_results["topic_words"]),
            "silhouette_score": float(silhouette),
            "total_documents": len(model_results["documents"]),
            "average_topic_size": statistics.mean(topic_sizes)     
        }

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r"))["model_topics"]


    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    doc_store_file = data_dir / "document_store.json"
    doc_store = InMemoryDocumentStore.load_from_disk(doc_store_file)

    topic_model = pipelines.TopicModelPipeline(
        doc_store, 
        umap_args=params["umap"], 
        hdbscan_args=params["hdbscan"],
        topic_merge_delta=params["topic_merge_delta"]
    )

    all_metrics = defaultdict(list)
    # run a number of replicates for umap and clustering
    for rep in range(params["reps"]):
        print(f"rep {rep}")
        topic_results = topic_model.run(arrow.utcnow().shift(days=-params["days"]))
        model_results = topic_results["topic_model"]
        metrics = evaluate_topics(model_results)
        print(metrics)
        all_metrics["total_topics"].append(metrics["total_topics"])
        all_metrics["silhouette_score"].append(metrics["silhouette_score"])
        all_metrics["total_documents"].append(metrics["total_documents"])

    def summary_stats(metric):
        return {
            "min": min(metric),
            "max": max(metric),
            "mean": statistics.mean(metric),
            "std": statistics.stdev(metric)
        }

    metrics_file = data_dir / "model_topics.json"
    with metrics_file.open("w") as fh:
        json.dump({
            "total_topics": summary_stats(all_metrics["total_topics"]),
            "silhouette_score": summary_stats(all_metrics["silhouette_score"]),
            "total_documents": statistics.mean(all_metrics["total_documents"])
        }, fh)