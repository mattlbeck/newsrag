from typing import List
import umap
import hdbscan
import numpy as np
from haystack import Document, component
from haystack.components.embedders import (
    HuggingFaceAPIDocumentEmbedder, SentenceTransformersDocumentEmbedder)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.ollama import OllamaGenerator
from top2vec import Top2Vec

DEFAULT_UMAP_ARGS = {'n_neighbors': 15,
                    'n_components': 5,
                    'metric': 'cosine'}

DEFAULT_HDBSCAN_ARGS = {'min_cluster_size': 15,
                        'metric': 'euclidean',
                        'cluster_selection_method': 'eom'}


@component
class JointEmbedderMixin:
    """Jointly embed documents along with individual words to form a vocabulary.
    
    The result is a set of documents that correspond to the embedded documents and additionally
    a set of embedded words.
    """

    def __init__(self, *args, min_word_count=3, ngram_vocab=False, **kwargs): 
        """
        :param min_word_count: 
            The minimum occurences of a word or phrase in documents for it to be used
            to describe topics.
        :param ngram_vocab: If True, use phrases within the word vocabulary
        """
        self.min_word_count = min_word_count
        self.ngram_vocab = ngram_vocab
        super(JointEmbedderMixin, self).__init__(*args, **kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        document_embeddings = super(JointEmbedderMixin, self).run(documents=documents)["documents"]

        from top2vec.top2vec import Top2Vec, default_tokenizer
        tokenized_corpus = [default_tokenizer(doc.content) for doc in documents]
        vocab = Top2Vec.get_label_vocabulary(tokenized_corpus, min_count=self.min_word_count, ngram_vocab=self.ngram_vocab, ngram_vocab_args=None)
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

    @component.output_types(documents=list[Document], topic_words=list[list[str]], umap_embedding=list)
    # using List over list for input types due to weird compat requirement from haystack
    def run(self, document_embeddings: List[Document], word_embeddings: List[Document]):
        """
        Compute topics and label documents with their assigned topic.

        This accepts the input document and word embeddings, runs umap and hdbscan
        to find topic clusters, describes these clusters using keywords, and then
        assigns a topic id to each document as well as the distance to its closest 
        topic.

        Outlier documents are also flagged. These are documents that are marked as 
        outlying by hdbscan. If they are flagged as an outlier, they are still assigned
        a topic id that represents their closest topic, as described by the topic score.

        Computing the input document and word embeddings can be done with an embedding
        component that inherits from JointEmbeddingMixin.

        :param document_embeddings: 
            a list of Documents representing the documents to find topics for. These
            documents will appear in the output with new topic-related metadata fields
            assigned. The documents must have been indexed with an associated embedding
        :param word_embeddings:
            a list of Documents representing the vocabulary that will be used to 
            described topics. The documents must have been indexed with an associated
            embedding.
        :return: 
            a dict of the following outputs:
                document_embeddings: the documents provided with input with new fields
                `topic_id` (int), `topic_score` (float), and `topic_outlier` (bool).
                topic_words: a list of words, or ngrams, that describe each topic
                umap_embedding: a list of embeddings (nd arrays) that are the umap
                projections of each document. Useful for downstream evaluation.

        """
        print(len(document_embeddings), " docs")
        print(len(word_embeddings), " words")
        self.documents = document_embeddings
        self.document_vectors = np.array([d.embedding for d in self.documents])

        self.vocab = [d.content for d in word_embeddings]
        self.word_vectors = [d.embedding for d in word_embeddings]

        # These computations are from `compute_topics` and have been surfaced here
        # in order to retain the umap embedding for further analysis
        umap_model = umap.UMAP(**self.umap_args).fit(self.document_vectors)
        umap_embedding = umap_model.embedding_

        cluster = hdbscan.HDBSCAN(**self.hdbscan_args).fit(umap_embedding)
        self.labels = cluster.labels_

        self._create_topic_vectors(self.labels)
        self._deduplicate_topics(topic_merge_delta=self.topic_merge_delta)
        self.topic_words, self.topic_word_scores = self._find_topic_words_and_scores(topic_vectors=self.topic_vectors)
        self.doc_top, self.doc_dist = self._calculate_documents_topic(self.topic_vectors,
                                                                      self.document_vectors,
                                                                      topic_index=None)
        
        # calculate topic sizes
        self.topic_sizes = self._calculate_topic_sizes(hierarchy=False)

        # re-order topics
        self._reorder_topics(hierarchy=False)
        

        
        topic_num, topic_score, _, _ = self.get_documents_topics(list(range(len(self.documents))))
        for num, score, doc, label in zip(topic_num, topic_score, self.documents, self.labels):
            doc.meta["topic_id"] = num
            doc.meta["topic_score"] = score
            # flag as an outlier if the original hdbscan label was -1
            doc.meta["topic_outlier"] = (label == -1)

        topic_words, word_scores, topic_nums = self.get_topics()
        return {"documents": self.documents, "topic_words": topic_words, "umap_embedding": umap_embedding}
        