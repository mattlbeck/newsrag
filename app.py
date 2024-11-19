import gradio as gr
from generator import Sources, StreamingText
from rag import RAG, RAGSummariser
import feeds
import generator


from multiprocessing.pool import ThreadPool
from topics import DocumentTopics
import re


with gr.Blocks() as demo:

    def topics():
        news = feeds.download_feeds((feeds.TheGuardian, feeds.AssociatedPress, feeds.BBC, feeds.ABC, feeds.CNBC, feeds.FoxNews, feeds.EuroNews ))
        print(f"{len(news)} news articles")
        model = DocumentTopics(news)
        model.run()
        
        return model, gr.update(choices=model.get_topic_descriptions(), value=None)


    def user(user_message, history:list):
        
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]
    
    def summarise(topic_model: DocumentTopics, topic_num: int):
        """This clears the chat history and starts again with a new news summary"""
        
        newsrag = RAGSummariser()

        topic_documents = topic_model.get_documents_for_topic(topic_num)

        async_result = newsrag.run_async(topic_documents)

        history = [{"role": "assistant", "content": ""}]
        # Run the news summarisation pipeline
        for content, sources in newsrag.stream_output(topic_documents):
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

    #text.submit(summarise, [text], [chatbot])
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
    demo.load(topics, outputs=[topic_model, topic_selection])
    topic_selection.select(summarise, [topic_model, topic_selection], [chatbot, sources])
demo.launch()