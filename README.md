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

## The App

These instructions should get you setup with the app on a local machine.

Prerequisites:

 - ollama with llama3.2:3B downloaded (or your preferred instruction-tuned LLM)

Install the repo package to your current environment, either via pip or poetry:

```
# using pip
pip install .
# using poetry
poetry install && poetry shell
```

Run the Gradio app

```
gradio app.py
```

You can alter change the generator and embedder that is used by editing the `config.yaml`.

## The Experiments

The repo also contains a DVC-orchestrated experiment pipeline for tuning the topic
modelling. The experiment scripts, split into checkpointed stages, are in `exp/` and 
tracked parameters that have undergone tuning can be found in `params.yaml`.

Typically, the head of the main branch in this repo will represent the best experimental 
setup, and the app will derive default parameters for topic modelling from `params.yaml`.
Promoting a DVC experiment to a new commit on the main branch and pushing this to 
the spaces hub represents deployment of the app to a prod environment.

### Running experiments

Ensure you have the experiments dependency group installed (`poetry install --group experiments`).
The run a DVC experiment with e.g.:

```
dvc exp run --temp --set-param model_topics.umap.n_neighbors=20
```