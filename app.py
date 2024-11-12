import gradio as gr
from rag import RAG, get_news
from threading import Thread
from queue import Queue

class StreamingText:

    def __init__(self):
        self._text = Queue()

    def __call__(self, text_chunk):

        self._text.put(text_chunk.content)

    def __iter__(self):
        return self
    
    def __next__(self):
        return self._text.get()



news = get_news("https://theguardian.com/uk/rss")
newsrag = RAG(documents=news)

def reply(message, history):
    streamer = StreamingText()
    newsrag.llm.streaming_callback = streamer
    t = Thread(target=newsrag.run, kwargs={"question": message})
    t.start()
    partial_message = ""
    for new_token in iter(streamer):
        if new_token != "<":
            partial_message += new_token
            yield partial_message

demo = gr.ChatInterface(fn=reply, type="messages", examples=["What are the latest UK government announcements?"], title="News RAG")
demo.launch()