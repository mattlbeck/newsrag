"""
DVC pipeline stage.
Downloads all available news feeds entries and writes them file.
"""
import newsrag.feeds as feeds
from pathlib import Path
import json
import jsonlines
from collections import defaultdict

if __name__ == "__main__":
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    docs = feeds.download_feeds()
    vendor_count = defaultdict(int)
    for doc in docs:
        for subfeed in doc.meta["subfeeds"]:
            vendor_count[doc.meta["vendor"] + "." + subfeed] += 1
    print(vendor_count)
    
    # serialize to json-lines
    out_file =  data_dir / "documents.jsonl"
    with jsonlines.Writer(out_file.open("w")) as writer:
        for doc in docs:
            writer.write(doc.to_dict())

    metrics_file = data_dir / "download_feeds.json"
    with metrics_file.open("w") as fh:
        json.dump({
            "total_documents": len(docs)
        }, fh)