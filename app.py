import gradio as gr
from rag import RAG, get_news
from multiprocessing.pool import ThreadPool
from queue import Queue

class StreamingText:

    def __init__(self):
        self._text = Queue()
        self._done = False

    def __call__(self, text_chunk):
        if text_chunk.meta["done"]:
            self._done = True
        self._text.put(text_chunk.content)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._done:
            raise StopIteration()
        return self._text.get(timeout=20)



news = get_news("https://theguardian.com/uk/rss")
newsrag = RAG(documents=news)



with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    text = gr.Textbox()
    prompt = gr.Textbox()
    state = gr.State()

    def user(user_message, history:list):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]


    def bot(history: list):
        if not history:
            return
        streamer = StreamingText()
        newsrag.llm.streaming_callback = streamer

        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(newsrag.run, kwds={"question": history[-1]["content"]})

        history.append({"role": "assistant", "content": ""})
        for new_token in iter(streamer):
            history[-1]["content"] += new_token
            yield history, ""
        
        pipeline_result = async_result.get()
        history[-1] = {"role": "assistant", "content": pipeline_result["llm"]["replies"][0]}
        yield history, pipeline_result["prompt_builder"]["prompt"]

    text.submit(user, [text, chatbot], [text, chatbot], queue=False).then(bot, chatbot, [chatbot, prompt])
demo.launch()