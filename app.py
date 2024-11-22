import gradio as gr
from generator import Sources, StreamingText
from rag import RAG, RAGSummariser, get_document_store, JointDocumentIndexingPipeline, TopicPipeline, DescribeTopicPipeline, TopicRetrievalPipeline
import feeds
import generator
import arrow

from multiprocessing.pool import ThreadPool
from topics import DocumentTopics
import re

document_store = get_document_store()

with gr.Blocks() as demo:

    def topics():
        """Downloads news feeds and indexes their content in to the central document store"""
        # download news from various feeds, formatted as haystack Document objects complete with some metadata
        news = feeds.download_feeds((feeds.TheGuardian, feeds.AssociatedPress, feeds.BBC, feeds.ABC, feeds.CNBC, feeds.FoxNews, feeds.EuroNews, feeds.TechCrunch, feeds.Wired, feeds.ArsTechnica))
        print(f"{len(news)} news articles")

        # Use the indexing pipeline to embed and write these documents to the chosen document store
        indexing = JointDocumentIndexingPipeline(document_store=document_store, min_word_count=0)
        indexing.run(news)

        # the topic pipeline discovers topics within the embedded documents and labels them with the embedded word vocabulary
        topics = TopicPipeline(document_store=document_store)
        result = topics.run(min_date=arrow.utcnow().shift(days=-1))

        # Describe each topic with a human readable title
        topic_describer = DescribeTopicPipeline()
        topic_descriptions = [topic_describer.run(topic) for topic in result["topic_model"]["topic_words"]]
        
        return gr.update(choices=topic_descriptions, value=None)


    def user(user_message, history:list):
        
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]
    
    def summarise(topic_num: int):
        """Summarises the given topic by retrieving documents related to that topic and 
        putting them throuth the summariser pipeline.
        
        This clears the chat history and starts again with a new news summary"""
        documents = TopicRetrievalPipeline(document_store=document_store, document_count=30).run(topic_id=topic_num)
        newsrag = RAGSummariser()

        async_result = newsrag.run_async(documents=documents)

        history = [{"role": "assistant", "content": ""}]
        # Run the news summarisation pipeline
        for content, sources in newsrag.stream_output(documents):
            history[0]["content"] = content
            bibliography = "\n".join(sources.generate_bibliography())
            yield history, bibliography

        pipeline_result = async_result.get()
        final_output = [{"role": "assistant", "content": pipeline_result["llm"]["replies"][0]}]
        history.append({"role": "user", "content": pipeline_result["prompt_builder"]["prompt"]})
        history.append({"role": "assistant", "content": pipeline_result["llm"]["replies"][0]})
        yield history,  "\n".join(sources.generate_bibliography())

    def bot(history: list):
        newsrag = RAG(documents=news)
        if not history:
            return
        streamer = StreamingText()
        newsrag.llm.streaming_callback = streamer

        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(newsrag.run, kwds={"question": history[-1]["content"]}, error_callback=lambda x: print("Error in generation thread: ", x))

        history.append({"role": "assistant", "content": ""})
        for new_token in iter(streamer):
            history[-1]["content"] += new_token
            yield history, ""
        
        pipeline_result = async_result.get()
        
        yield history, pipeline_result["prompt_builder"]["prompt"]

    with gr.Row():
        title = gr.Markdown("# newsrag")
    with gr.Row():
        with gr.Column():
            topic_selection = gr.Dropdown(label="Select a topic", type="index")
            topic_model = gr.State()
            chatbot = gr.Chatbot(type="messages", height=600)
        with gr.Column():
            sources = gr.Markdown("Sources go here", container=True, height=800)
    topic_model = gr.State()
    demo.load(topics, outputs=[topic_selection])
    topic_selection.select(summarise, [topic_selection], [chatbot, sources])
demo.launch()
