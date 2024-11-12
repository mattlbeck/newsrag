import os
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
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

def get_embedding_retriever(docs):
    doc_embedder = SentenceTransformersDocumentEmbedder()
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run([Document(content=d) for d in docs])["documents"]

    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs_with_embeddings)
    retriever = InMemoryEmbeddingRetriever(doc_store)
    return retriever

class RAG:
    # Build a RAG pipeline
    prompt_template = """
    Given these news article summaries, answer the question.
    Article Summaries:
    {% for doc in documents %}
        ARTICLE START
        {{ doc.content }}
        ARTICLE END
    {% endfor %}
    Question: {{question}}
    Answer:
    """

    def __init__(self, documents, top_k=5, embeddings=True, streaming_callback=None):
        self.embeddings = embeddings
        self.pipeline = Pipeline()

        # set up retriever with a sentence embedding
        self.retriever = get_embedding_retriever(documents)
        self.embedder = SentenceTransformersTextEmbedder()
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("embedder", self.embedder)
       

        self.prompt_builder = PromptBuilder(template=self.prompt_template)
        self.llm = OllamaGenerator(model="llama3.2", streaming_callback=streaming_callback)
        
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        
        self.pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "prompt_builder")
        self.pipeline.connect("prompt_builder.prompt", "llm")
    
    def run(self, question):
        # Ask a question
        results = self.pipeline.run(
            {
                "embedder": {"text": question},
                "prompt_builder": {"question": question},
            }, include_outputs_from=("retriever", "prompt_builder", "llm")
        )
        return results


if __name__ == "__main__":
    news = get_news("https://theguardian.com/uk/rss")
    newsrag = RAG(documents=news)

    print(newsrag.run("What did Trump do this time?")["llm"]["replies"][0])