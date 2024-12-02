---
title: newsrag
emoji: 👀
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Topic discovery and summarisation of news feeds.
---
# NewsRAG

This is a gradio-based demonstration of a haystack-based RAG app that ingests RSS news feeds and then discovers topics, summarises, and allows free-form QA on the article headlines.

## Get Started

These instructions should get you setup with the app on a local machine.

Prerequisites:

 - ollama with llama3.2:3B downloaded (or your preferred instruction-tuned LLM)

Install requirements into your favourite virtual environment

```
pip install -r requirements.txt
```

Run the Gradio app

```
gradio app.py
```

You can alter change the generator and embedder that is used by editing the `config.yaml`.