import newsrag.feeds as feeds
from pathlib import Path
import json
import arrow

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.rankers import MetaFieldRanker

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
    outlier_mask = np.array([d.meta["topic_outlier"] for d in documents])

    # assess documents that are not outliers
    topic_ids = [d.meta["topic_id"] for d in documents if not d.meta["topic_outlier"]]
    topic_sizes = Counter(topic_ids).values()
    umap_embeddings = model_results["umap_embedding"][~outlier_mask]

    num_outliers = np.sum(outlier_mask)
    silhouette = silhouette_score(umap_embeddings, labels=topic_ids)
    return {
            "total_topics": len(model_results["topic_words"]),
            "silhouette_score": float(silhouette),
            "total_documents": len(model_results["documents"]),
            "average_topic_size": statistics.mean(topic_sizes),
            "total_outliers": int(num_outliers)
        }

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r"))["model_topics"]


    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    doc_store_file = data_dir / "document_store.json"
    doc_store = InMemoryDocumentStore.load_from_disk(doc_store_file)
    ranker = MetaFieldRanker(meta_field="timestamp", missing_meta="drop", top_k=1)

    # get the newest document in the store
    latest_doc = ranker.run(doc_store.filter_documents())["documents"][0]
    latest_date = arrow.get(latest_doc.meta["timestamp"])
    print(f"Latest docment: {latest_date}")
    

    topic_model = pipelines.TopicModelPipeline(
        doc_store, 
        umap_args=params["umap"], 
        hdbscan_args=params["hdbscan"],
        topic_merge_delta=params["topic_merge_delta"]
    )

    all_metrics = defaultdict(list)
    # run a number of replicates for umap and clustering to assess the stability 
    # of the model
    for rep in range(params["reps"]):
        print(f"rep {rep}")

        # model on topics with min date relative to the latest document date
        topic_results = topic_model.run(latest_date.shift(days=-params["days"]))
        model_results = topic_results["topic_model"]
        metrics = evaluate_topics(model_results)
        print(metrics)
        all_metrics["total_topics"].append(metrics["total_topics"])
        all_metrics["silhouette_score"].append(metrics["silhouette_score"])
        all_metrics["total_documents"].append(metrics["total_documents"])
        all_metrics["total_outliers"].append(metrics["total_outliers"])

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
            "total_documents": statistics.mean(all_metrics["total_documents"]),
            "total_outliers": statistics.mean(all_metrics["total_outliers"])
        }, fh)