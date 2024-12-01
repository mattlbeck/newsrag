"""This script updates the publicly shared hub data with new news feeds"""
import datasets
import datasets.data_files
from newsrag import feeds
from functools import partial

repo_name = "mattlbeck/newsfeeds"

if __name__ == "__main__":    

    fields = ("title", "content", "link", "published", "timestamp", "vendor", "subfeeds")

    documents = feeds.download_feeds()
    # latest document is the last document, which makes adding updates easiers
    documents.sort(key=lambda x: x.meta["timestamp"])

    def gen_records(documents):
        for doc in documents:
            record = doc.to_dict()
            yield {key: record[key] for key in fields}

    try:
        dataset = datasets.load_dataset(repo_name)
        latest_timestamp = dataset["train"][-1]["timestamp"]

        # efficiently identify newer articles
        for i, doc in enumerate(documents):
            if doc.meta["timestamp"] > latest_timestamp:
                break
        new_docs = documents[i:]
        new_data = datasets.Dataset.from_generator(partial(gen_records, new_docs))

        dataset = datasets.concatenate_datasets([dataset["train"], new_data])
        print(f"Uploading {len(dataset)} records of new data")
        
    except datasets.data_files.EmptyDatasetError:
        dataset = datasets.Dataset.from_generator(partial(gen_records, documents))
        print(f"Uploading initial dataset of {len(dataset)} records")

    dataset.push_to_hub(repo_name)

