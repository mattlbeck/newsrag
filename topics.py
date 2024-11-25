from top2vec import Top2Vec
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack import component
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, HuggingFaceAPIDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
import numpy as np
from typing import List
    
@component
class JointEmbedderMixin:
    """Jointly embed documents along with individual words to form a vocabulary.
    
    The result is a set of documents that correspond to the embedded documents and additionally
    a set of embedded words.
    """

    def __init__(self, *args, min_word_count=3, **kwargs): 
        self.min_word_count=min_word_count
        super(JointEmbedderMixin, self).__init__(*args, **kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        document_embeddings = super(JointEmbedderMixin, self).run(documents=documents)["documents"]

        from top2vec.top2vec import Top2Vec, default_tokenizer
        tokenized_corpus = [default_tokenizer(doc.content) for doc in documents]
        vocab = Top2Vec.get_label_vocabulary(tokenized_corpus, min_count=self.min_word_count, ngram_vocab=False, ngram_vocab_args=None)
        vocab_docs = [Document(content=v) for v in vocab]

        word_embeddings = super(JointEmbedderMixin, self).run(documents=vocab_docs)["documents"]
        all_documents = []
        for doc in document_embeddings:
            doc.meta["type"] = "document"
            all_documents.append(doc)

        for word in word_embeddings:
            word.meta["type"] = "word"
            all_documents.append(word)
        return {"documents": all_documents}
    

class SentenceTransformersJointEmbedder(JointEmbedderMixin, SentenceTransformersDocumentEmbedder):
    """Uses a sentence transformer as an embedder but additonally embeds a vocabulary of words as
    another set of documents."""

class HuggingfaceAPIJointEmbedder(JointEmbedderMixin, HuggingFaceAPIDocumentEmbedder):
    """Uses the huggingface API the embedder but additionally embeds a vocabulary of words as another
    set of documents"""
        
@component        
class TopicModel(Top2Vec):
    """
    Custom haystack component that uses Top2Vec to discover topics from a set of documents and 
    related vocabularly. The documents are updated with metadata on their assigned topics.

    The component outputs: the documents with additional topic metadata, and a list of topics and
    topic keywords to describe them.
    """
    def __init__(self,
                 c_top2vec_smoothing_window=5,
                 topic_merge_delta=0.1,
                 umap_args=None,
                 gpu_umap=False,
                 hdbscan_args=None,
                 gpu_hdbscan=False,
                 index_topics=False,
                 ):
        self.c_top2vec_smoothing_window = c_top2vec_smoothing_window
        self.topic_merge_delta = topic_merge_delta
        self.gpu_umap = gpu_umap
        self.gpu_hdbscan = gpu_hdbscan
        self.index_topics = index_topics

        # initialize topic indexing variables
        self.topic_index = None
        self.serialized_topic_index = None
        self.topics_indexed = False

        # initialize document indexing variables
        self.document_index = None
        self.serialized_document_index = None
        self.documents_indexed = False
        self.index_id2doc_id = None
        self.doc_id2index_id = None

        # initialize word indexing variables
        self.word_index = None
        self.serialized_word_index = None
        self.words_indexed = False
        
        # required attribute in compute_topics
        self.contextual_top2vec = False
        self.document_ids = None

        self.umap_args = {
                    'n_neighbors': 50,
                    'n_components': 5,
                    'metric': 'euclidean'}
        if umap_args is not None:
            self.umap_args.update(umap_args)
        self.hdbscan_args = {'min_cluster_size': 15}
        if hdbscan_args is not None:
            self.hdbscan_args.update(hdbscan_args)

    @component.output_types(documents=list[Document], topic_words=list[list[str]])
    # using List over list for input types due to weird compat requirement from haystack
    def run(self, document_embeddings: List[Document], word_embeddings: List[Document]):
        print(len(document_embeddings), " docs")
        print(len(word_embeddings), " words")
        self.documents = document_embeddings
        self.document_vectors = np.array([d.embedding for d in self.documents])

        self.vocab = [d.content for d in word_embeddings]
        self.word_vectors = [d.embedding for d in word_embeddings]

        self.compute_topics(umap_args=self.umap_args,
                            hdbscan_args=self.hdbscan_args,
                            topic_merge_delta=self.topic_merge_delta,
                            gpu_umap=self.gpu_umap,
                            gpu_hdbscan=self.gpu_hdbscan,
                            index_topics=self.index_topics,
                            contextual_top2vec=False,
                            c_top2vec_smoothing_window=self.c_top2vec_smoothing_window)
        
        topic_num, topic_score, _, _ = self.get_documents_topics(list(range(len(self.documents))))
        for num, score, doc in zip(topic_num, topic_score, self.documents):
            doc.meta["topic_id"] = num
            doc.meta["topic_score"] = score

        topic_words, word_scores, topic_nums = self.get_topics()
        return {"documents": self.documents, "topic_words": topic_words}
        