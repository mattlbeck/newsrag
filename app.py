import gradio as gr
from newsrag.generator import Sources
from newsrag.pipelines import SummarisationPipeline, JointDocumentIndexingPipeline, TopicModelPipeline, DescribeTopicPipeline, TopicRetrievalPipeline, QAGeneratorPipeline, QARetrievalPipeline
import newsrag.feeds as feeds
import arrow
from newsrag.config import AppConfig
from pathlib import Path
import jsonlines
from haystack import Document
import os

description = """
# newsrag
**Summarisation of daily news topics**.
 - top2vec discovers topics from the last day's news headlines across a diverse number of news vendors
 - An LLM summarises each topic, building a list of citations as the response is streamed
 - Q&A allows you to ask questions against the database of news headlines in a classic RAG paradigm
"""
config = AppConfig()


def get_bibliography(sources: Sources):
    source_list = []
    for i, source in enumerate(sources._sources):
        source_list.append(f"{i+1}. {source.meta['title']} - [{source.meta['vendor']}]({source.meta['link']})")
    return "\n".join(source_list)

def get_cached_news():
    """
    Loads and returns cached news from file if the file exists.
    The file is determined by an environment variable
    """
    cache_file = Path(os.environ["APP_DOCUMENT_CACHE"])
    if cache_file.exists():
        print(f"Loading news from cache {cache_file}")
        news = []
        with jsonlines.open(cache_file) as reader:
            for doc in reader:
                news.append(Document.from_dict(doc))
        return news


with gr.Blocks() as demo:

    def topics(document_store):
        """Downloads news feeds and indexes their content in to the central document store"""
        # download news from various feeds, formatted as haystack Document objects complete with some metadata
        news = get_cached_news()
        if not news: 
            print("Downloading fresh news")
            news = feeds.download_feeds()
            print(f"{len(news)} news articles")

        # Use the indexing pipeline to embed and write these documents to the chosen document store
        indexing = JointDocumentIndexingPipeline(document_store=document_store, joint_embedder=config.get_joint_document_embedder(min_word_count=3), min_word_count=0)
        indexing.run(news)

        # the topic pipeline discovers topics within the embedded documents and labels them with the embedded word vocabulary
        topics = TopicModelPipeline(document_store=document_store)
        result = topics.run(min_date=arrow.utcnow().shift(days=-1))

        # Describe each topic with a human readable title
        topic_describer = DescribeTopicPipeline(generator=config.get_generator_model())
        topic_descriptions = [topic_describer.run(topic) for topic in result["topic_model"]["topic_words"]]
        
        return gr.update(choices=topic_descriptions, value=None)
    
    def summarise(document_store, sources, topic_num: int):
        """Summarises the given topic by retrieving documents related to that topic and 
        putting them throuth the summariser pipeline.
        
        This clears the chat history and starts again with a new news summary"""
        documents = TopicRetrievalPipeline(document_store=document_store, document_count=30).run(topic_id=topic_num)
        newsrag = SummarisationPipeline(generator=config.get_generator_model())

        async_result = newsrag.run_async(documents=documents)

        history = [{"role": "assistant", "content": ""}]
        # Run the news summarisation pipeline
        for content, sources in newsrag.stream_output(documents, sources):
            history[0]["content"] = content
            bibliography = get_bibliography(sources)
            yield history, bibliography

        pipeline_result = async_result.get()
        # final_output = [{"role": "assistant", "content": pipeline_result["llm"]["replies"][0]}]
        # history.append({"role": "user", "content": pipeline_result["prompt_builder"]["prompt"]})
        # history.append({"role": "assistant", "content": pipeline_result["llm"]["replies"][0]})
        yield history,  get_bibliography(sources)

    def user_query(user_message, history:list):    
        if history is None:
            history = []
        return history + [{"role": "user", "content": user_message}], gr.update(value="")
    
    def qa(document_store, sources, history: list):
        retriever = QARetrievalPipeline(document_store=document_store, text_embedder=config.get_text_embedder())
        
        # TODO: additional conversation context
        question = history[-1]["content"]
        documents = retriever.run(question)
        print("retrieved", len(documents), "documents")
        qa = QAGeneratorPipeline(generator=config.get_generator_model())
        
        async_result = qa.run_async(question=question, documents=documents)

        history.append({"role": "assistant", "content": ""})
        # Run the news summarisation pipeline
        for content, sources in qa.stream_output(documents, sources=sources):
            history[-1]["content"] = content
            bibliography = get_bibliography(sources)
            yield history, bibliography
        
        yield history, bibliography

    ########
    # App UI
    ########
    # keep persistent document store and sources list
    sources = gr.State(value=Sources())
    document_store = gr.State(value=config.get_document_store())

    # arrange UI elements
    with gr.Row():
        with gr.Column():
            title = gr.Markdown(description)
            topic_selection = gr.Dropdown(label="Select a topic", type="index")
            qa_input = gr.Textbox(label="Ask a Question:")
            bibliography = gr.Markdown(label="Bibliography", container=True, height=400)
            
        with gr.Column():
            chatbot = gr.Chatbot(type="messages", height=800)
            
    
    # set actions and triggers
    demo.load(topics, inputs=[document_store], outputs=[topic_selection])
    topic_selection.select(summarise, inputs=[document_store, sources, topic_selection], outputs=[chatbot, bibliography])
    qa_input.submit(user_query, inputs=[qa_input, chatbot], outputs=[chatbot, qa_input]).then(qa, inputs=[document_store, sources, chatbot], outputs=[chatbot, bibliography])
demo.launch()
