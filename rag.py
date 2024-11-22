import os
from haystack import Pipeline, Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.writers.document_writer import DocumentWriter
from haystack.components.rankers import MetaFieldRanker
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from functools import cached_property
import arrow
from datetime import datetime
from multiprocessing.pool import ThreadPool
from topics import SentenceTransformersJointEmbedder, TopicModel
from haystack.components.routers import MetadataRouter

import generator

from haystack.components.retrievers import FilterRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack import Document


def get_document_store():
    document_store = InMemoryDocumentStore()
    return document_store




class StreamingGeneratorMixin:
    """Defines functions that provide async streamed results from an LLM component.
    
    These methods assume that there is an `llm` component as an attribute and a `run` method
    is defined.
    """
    def run_async(self, **run_kwargs):
        streamer = generator.StreamingText()
        self.llm.streaming_callback = streamer

        pool = ThreadPool(processes=1)
  
        async_result = pool.apply_async(self.run, kwds=run_kwargs, error_callback=lambda x: print("Error in generation thread: ", x))
        return async_result

    
    def stream_output(self, documents: list[Document]):
    
        # Stream the summary output, transforming citations
        # on the fly and building a bibliography to output to a second
        # component
        yield from generator.stream_sourced_output(iter(self.llm.streaming_callback), documents)


class JointDocumentIndexingPipeline:
    """
    Jointly index documents along with the document vocabulary
    """
    def __init__(self, document_store, min_word_count=3):
        self._store = document_store
        self.embedder = SentenceTransformersJointEmbedder(min_word_count=min_word_count)
        self.writer = DocumentWriter(document_store=document_store)
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", self.embedder)
        self.pipeline.add_component("writer", self.writer)
        self.pipeline.connect("embedder", "writer")

    def run(self, documents: list[Document]):
        return self.pipeline.run({"embedder": {"documents": documents}})
        

class TopicModelPipeline:
    """Discovers topics from documents within a document store.
    
    Documents used in the topic discovery are updated with metadata to identify their
    topic and topic score.
    """


    def __init__(self, document_store):
        self._store = document_store
        
        self.retriever = FilterRetriever(document_store)
        self.topic_model = TopicModel()

        # the router splits out documents that are docs form those that are the vocabulary to pass
        # as separate inputs to the TopicModel
        self.router = MetadataRouter(rules={
            "document_embeddings": {"field": "meta.type", "operator": "==", "value": "document"},
            "word_embeddings": {"field": "meta.type", "operator": "==", "value": "word"}
            })
        
        # Documents are written back to the document store once they are updated with their topic
        # numbers and score
        self.writer = DocumentWriter(document_store, policy=DuplicatePolicy.OVERWRITE)

        self.pipeline = Pipeline()
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("topic_model", self.topic_model)
        self.pipeline.add_component("router", self.router)
        self.pipeline.add_component("writer", self.writer)
        self.pipeline.connect("retriever", "router")
        self.pipeline.connect("router.document_embeddings", "topic_model.document_embeddings")
        self.pipeline.connect("router.word_embeddings", "topic_model.word_embeddings")
        self.pipeline.connect("topic_model.documents", "writer")


    def run(self, min_date: datetime):
        return self.pipeline.run({
            "retriever": {"filters": {
                          "operator": "OR",
                          "conditions": [
                            {"field": "meta.date", "operator": ">", "value": min_date.timestamp() },
                            {"field": "meta.type", "operator": "==", "value": "word"}
                          ]}}
        }, include_outputs_from="topic_model")


class DescribeTopicPipeline:
    prompt_template = """
Below is a list of keywords derived from various news articles that share the same topic. Please provide a short description, maximum 5 words, of the topic that best fits. Output only the topic description.
Keywords: {{ topic_words|join(', ') }}
Topic: 
"""

    def __init__(self, max_words=10):
        self.max_words=max_words

        self.prompt = PromptBuilder(self.prompt_template)
        self.llm = OllamaGenerator(model="llama3.2")
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt", self.prompt)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt", "llm")

    def run(self, topic_words: list[str]):
        return self.pipeline.run({ "prompt": {"topic_words": topic_words}})["llm"]["replies"][0]

def get_embedding_retriever(docs, top_k):
    doc_embedder = SentenceTransformersDocumentEmbedder()
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)["documents"]

    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs_with_embeddings)
    retriever = InMemoryEmbeddingRetriever(doc_store, top_k=top_k)
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

    def __init__(self, documents: list[Document], top_k=5, embeddings=True, streaming_callback=None):
        self.embeddings = embeddings
        self.pipeline = Pipeline()

        # set up retriever with a sentence embedding
        self.retriever = get_embedding_retriever(documents, top_k)
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
        min_date = arrow.utcnow().shift(days=-1)
        results = self.pipeline.run(
            {
                "embedder": {"text": question},
                "prompt_builder": {"question": question},
                "retriever": {"filters": {"field": "meta.date", "operator": ">", "value": min_date }}
            }, include_outputs_from=("retriever", "prompt_builder", "llm")
        )
        return results
    
class TopicRetrievalPipeline:
    """Retrieve documents from a document score that belong to a specific topic.
    
    A ranker will additionally order the topics and limit the number of documents returned.
    """

    def __init__(self, document_store, document_count=10):
        self.document_store = document_store
        self.retriever = FilterRetriever(document_store=document_store)
        self.ranker = MetaFieldRanker(meta_field="topic_score", missing_meta="drop", top_k=document_count)
        self.pipeline = Pipeline()
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("ranker", self.ranker)
        self.pipeline.connect("retriever", "ranker")

    def run(self, topic_id):
        results = self.pipeline.run(
            {
                "retriever": {"filters": {"field": "meta.topic_id", "operator": "==", "value": topic_id}}
            }
        )
        return results["ranker"]["documents"]


class RAGSummariser(StreamingGeneratorMixin):

    prompt_template = """You will be provided with a list of news articles from today. Write a few paragraphs that summarises the news. Do not refer to the existance of the news articles themselves, their titles, or their formatting. After each statement, provide one or more citations in the form "[ARTICLE 1]", where ARTICLE 1 corresponds to the identifier of the article from which you sourced your statement. You may source a statement from more than one article, for example "[ARTICLE 1, ARTICLE 2]". Place these citations within the sentence e.g. "This is a statement [ARTICLE 1]."

News articles:
{% for doc in documents %}
ARTICLE {{ loop.index }}: {{ doc.content }}
{% endfor %}

Summary:"""
    def __init__(self):
        self.prompt_builder = PromptBuilder(template=self.prompt_template)
        self.llm = OllamaGenerator(model="llama3.2", generation_kwargs={"num_ctx": 4096})

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt_builder.prompt", "llm")
    
    def run(self, documents: list[Document]):
        results = self.pipeline.run(
            {
                "prompt_builder": {"documents": documents}
            },
            include_outputs_from=["prompt_builder"]
        )
        return results



if __name__ == "__main__":
    news = get_news("https://theguardian.com/uk/rss")
    newsrag = RAG(documents=news)

    print(newsrag.run("What did Trump do this time?")["llm"]["replies"][0])