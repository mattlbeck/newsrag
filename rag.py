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
import arrow


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
        results = self.pipeline.run(
            {
                "embedder": {"text": question},
                "prompt_builder": {"question": question},
                "retriever": {"filters": {"field": "meta.date", "operator": ">", "value": min_date }}
            }, include_outputs_from=("retriever", "prompt_builder", "llm")
        )
        print(results)
        return results
    

class RAGSummariser(RAG):

    prompt_template = """
        You will be provided with a list of news articles from today. Write a few paragraphs that summarises selected events. Do not refer to the existance of the news articles themselves, their titles, or their formatting. With each statement you provide, leave 1 or more citations in the form of [ARTICLE NUMBER], where ARTICLE NUMBER corresponds to the number of the articles from which you sourced that statement. You may source a statement from more than one article and site them as [ARTICLE 1, ARTICLE 2, etc].

        News articles:
        {% for doc in documents %}
            ARTICLE {{ loop.index }}: {{ doc.content }}
        {% endfor %}
        """
    def __init__(self):
        self.prompt_builder = PromptBuilder(template=self.prompt_template)
        self.llm = OllamaGenerator(model="llama3.2")

        self.pipeline = Pipeline()
        
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        
        self.pipeline.connect("prompt_builder.prompt", "llm")
    
    def run(self, documents):
        min_date = arrow.utcnow().shift(days=-1)
        # Ask a question
        results = self.pipeline.run(
            {
                "prompt_builder": {"documents": documents}
            }, 
            include_outputs_from=("prompt_builder", "llm")
        )
        return results


if __name__ == "__main__":
    news = get_news("https://theguardian.com/uk/rss")
    newsrag = RAG(documents=news)

    print(newsrag.run("What did Trump do this time?")["llm"]["replies"][0])