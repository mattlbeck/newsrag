import os
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from functools import cached_property

import feedparser

def get_news(news_rss):
    d = feedparser.parse(news_rss)
    summaries = [x["summary"] for x in d["entries"]]
    return summaries

def load_documents(docs):
    # Write documents to InMemoryDocumentStore
    document_store = InMemoryDocumentStore()
    document_store.write_documents([
        Document(content=d) for d in docs
    ])
    return document_store

class RAG:
    # Build a RAG pipeline
    prompt_template = """
    Given these news article summaries, answer the question.
    Article Summaries:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Question: {{question}}
    Answer:
    """

    def __init__(self, documents, top_k=5):
        self.retriever = InMemoryBM25Retriever(document_store=load_documents(documents), top_k=top_k)
        self.prompt_builder = PromptBuilder(template=self.prompt_template)
        self.llm = OllamaGenerator(model="llama3.2")

    @cached_property
    def pipeline(self):
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("retriever", self.retriever)
        rag_pipeline.add_component("prompt_builder", self.prompt_builder)
        rag_pipeline.add_component("llm", self.llm)
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "llm")
        return rag_pipeline
    
    def run(self, question):
        # Ask a question
        results = self.pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
            }, include_outputs_from=("retriever", "prompt_builder", "llm")
        )
        return results

