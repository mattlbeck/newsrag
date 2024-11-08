import gradio as gr
from rag import RAG, get_news

news = get_news("https://theguardian.com/uk/rss")
newsrag = RAG(documents=news)

def reply(message, history):
    return newsrag.run(message)["llm"]["replies"][0]

demo = gr.ChatInterface(fn=reply, type="messages", examples=["How are the Democrats taking the result of the US election?"], title="Echo Bot")
demo.launch()