from top2vec import Top2Vec
from haystack_integrations.components.generators.ollama import OllamaGenerator

class DocumentTopics:
    """Builds and holds a topic model along with the input documents"""

    def __init__(self, documents):
        self.documents = documents
        self.model = None

    def run(self):
        """Run model on the documents"""

        # temp workaround for a bug on this attribute
        Top2Vec.contextual_top2vec = False
        self.model = Top2Vec(
            [d.content for d in self.documents], 
            embedding_model='universal-sentence-encoder', 
            min_count=1, 
            speed="deep-learn", 
            ngram_vocab=False,
            keep_documents=True
        )

    def get_topic_descriptions(self):
        """
        Runs topic keywords through an LLM to produce readable descriptions for
        each
        """
        num_topics = self.model.get_num_topics()
        topic_words, word_scores, topic_nums = self.model.get_topics(num_topics)
        topic_descriptions = []
        for topic in topic_words:
            topic_descriptions.append(describe_topic(topic))
        return topic_descriptions

    def get_documents_for_topic(self, topic_num: int):
        """
        Retrieves documents that belong to the given topic.
        """
        _, _, document_ids = self.model.search_documents_by_topic(topic_num=topic_num, num_docs=10)
        
        documents = []
        for doc_id in document_ids:
            documents.append(self.documents[doc_id])
        return documents
            


def describe_topic(topic_words):
    llm = OllamaGenerator(model="llama3.2")
    prompt = f"""
    Below is a list of keywords that all derive from one topic. Please provide a short description, maximum 5 words, of the topic that best fits. Output only the topic description.
    Keywords: {topic_words[:10]}
    Topic: 
    """
    return llm.run(prompt=prompt)["replies"][0]