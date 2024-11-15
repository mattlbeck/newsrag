import gradio as gr
from rag import RAG, RAGSummariser
from feeds import TheGuardian, AssociatedPress, BBC, download_feeds
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
        return self._text.get()



news = download_feeds((TheGuardian, AssociatedPress, BBC))
newsrag = RAGSummariser(documents=news)



with gr.Blocks() as demo:
    text = gr.Textbox(label="Summarise news about...")

    chatbot = gr.Chatbot(type="messages")
   
    query = gr.Textbox()
    state = gr.State()

    def user(user_message, history:list):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    def summarise(topic: str):
        """This clears the chat history and starts again with a new news summary"""
        streamer = StreamingText()
        newsrag.llm.streaming_callback = streamer

        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(newsrag.run, kwds={"question": topic}, error_callback=lambda x: print("Error in generation thread: ", x))

        history = [{"role": "assistant", "content": ""}]
        for new_token in iter(streamer):
            history[0]["content"] += new_token
            yield history
        
        pipeline_result = async_result.get()
        yield [{"role": "assistant", "content": pipeline_result["llm"]["replies"][0]}]

    def bot(history: list):
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
        history[-1] = {"role": "assistant", "content": pipeline_result["llm"]["replies"][0]}
        yield history, pipeline_result["prompt_builder"]["prompt"]

    text.submit(summarise, [text], [chatbot])
demo.launch()