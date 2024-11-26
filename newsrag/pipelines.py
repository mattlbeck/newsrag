from datetime import datetime
from multiprocessing.pool import ThreadPool

import arrow
import newsrag.generator as generator
from haystack import Document, Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.rankers import MetaFieldRanker
from haystack.components.retrievers import FilterRetriever
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.routers import MetadataRouter
from haystack.components.writers.document_writer import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy

from newsrag.topics import (JointEmbedderMixin,
                            SentenceTransformersJointEmbedder, TopicModel)


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

    
    def stream_output(self, documents: list[Document], sources: generator.Sources):
    
        # Stream the summary output, transforming citations
        # on the fly and building a bibliography to output to a second
        # component
        yield from generator.stream_sourced_output(iter(self.llm.streaming_callback), sources, documents)


class JointDocumentIndexingPipeline:
    """
    Jointly index documents along with the document vocabulary
    """
    def __init__(self, document_store, joint_embedder: JointEmbedderMixin, min_word_count=3):
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


    def __init__(self, document_store, **top2vec_args):
        self._store = document_store
        
        self.retriever = FilterRetriever(document_store)
        self.topic_model = TopicModel(**top2vec_args)

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

    def __init__(self, generator, max_words=10):
        self.max_words=max_words

        self.prompt = ChatPromptBuilder([ChatMessage.from_user(self.prompt_template)])
        self.llm = generator
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt", self.prompt)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt", "llm")

    def run(self, topic_words: list[str], debug=False):
        result = self.pipeline.run({ "prompt": {"topic_words": topic_words}}, include_outputs_from=["prompt"])
        if debug:
            return result
        return result["llm"]["replies"][0].content
    

class QARetrievalPipeline:
    """Retrieve documents from a document score relevant to the query.
    
    A ranker will additionally order the topics and limit the number of documents returned.
    """

    def __init__(self, document_store, text_embedder, document_count=10):
        self.document_store = document_store
        self.retriever = InMemoryEmbeddingRetriever(document_store, top_k=document_count)
        self.embedder = text_embedder

        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", self.embedder)
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.connect("embedder", "retriever")

    def run(self, query) -> list[Document]:
        results = self.pipeline.run(
            {
                "embedder": {"text": query},
                "retriever": {"filters": {"field": "meta.type", "operator": "==", "value": "document"}}
            }
        )
        return results["retriever"]["documents"]
    
class QAGeneratorPipeline(StreamingGeneratorMixin):
    """Classic QA pipeline over documents in the given document store"""
    prompt_template = """You will be provided with a list of news articles from today. Based on these articles, answer the user's question. Do not refer to the existance of the news articles themselves, their titles, or their formatting. After each statement, provide one or more citations in the form "[ARTICLE 1]", where ARTICLE 1 corresponds to the identifier of the article from which you sourced your statement. You may source a statement from more than one article, for example "[ARTICLE 1, ARTICLE 2]". Place these citations within the sentence e.g. "This is a statement [ARTICLE 1]."

News articles:
{% for doc in documents %}
ARTICLE {{ loop.index }}: {{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer
"""

    def __init__(self, generator):

        self.prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(self.prompt_template)])
        self.llm = generator
        
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        
        self.pipeline.connect("prompt_builder.prompt", "llm")
    
    def run(self, question: str, documents: list[Document], min_date: datetime=None):
        if not min_date:
            min_date = arrow.utcnow().shift(days=-1)
        results = self.pipeline.run(
            {
                "prompt_builder": {"question": question, "documents": documents}
            }
        )
        return results["llm"]["replies"][0].content
    
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


class SummarisationPipeline(StreamingGeneratorMixin):

    prompt_template = """You will be provided with a list of news articles from today. Write a few paragraphs that summarises the news. Do not refer to the existance of the news articles themselves, their titles, or their formatting. After each statement, provide one or more citations in the form "[ARTICLE 1]", where ARTICLE 1 corresponds to the identifier of the article from which you sourced your statement. You may source a statement from more than one article, for example "[ARTICLE 1, ARTICLE 2]". Place these citations within the sentence e.g. "This is a statement [ARTICLE 1]."

News articles:
{% for doc in documents %}
ARTICLE {{ loop.index }}: {{ doc.content }}
{% endfor %}

Summary:"""
    def __init__(self, generator):
        self.prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(self.prompt_template)])
        self.llm = generator

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt_builder", "llm")
    
    def run(self, documents: list[Document], debug=False):
        results = self.pipeline.run(
            {
                "prompt_builder": {"documents": documents}
            },
            include_outputs_from=["prompt_builder"]
        )
        if debug:
            return results
        return results["llm"]["replies"][0].content

class ArticleSegmentRetrievalPipeline:
    """
    Given a query, retrieve articled that may contain info on
    the query based on headlines, then download the articles and rank
    the segments, returning the top k segments to use in answer generation

    This is TODO at the moment
    """