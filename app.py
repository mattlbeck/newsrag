import gradio as gr
from rag import RAG, RAGSummariser
from feeds import TheGuardian, AssociatedPress, BBC, download_feeds
from multiprocessing.pool import ThreadPool
from queue import Queue
from top2vec import Top2Vec
from haystack import Document

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



with gr.Blocks() as demo:
    text = gr.Textbox(label="Summarise news about...")
    topic_selection = gr.Dropdown(label="Select a topic", type="index")
    topic_model = gr.State()

    chatbot = gr.Chatbot(type="messages")
   
    query = gr.Textbox()
    topic_model = gr.State()

    def topics(topic_selection):
        Top2Vec.contextual_top2vec = False
        model = Top2Vec(
            [d.content for d in news], 
            embedding_model='universal-sentence-encoder', 
            min_count=1, 
            speed="deep-learn", 
            ngram_vocab=False,
            keep_documents=True
        )
        num_topics = model.get_num_topics()
        topic_words, word_scores, topic_nums = model.get_topics(num_topics)
        
        return model, gr.update(choices=[", ".join(words[:5]) for words in topic_words], value=None)


    def user(user_message, history:list):
        
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]
    
    def summarise(topic_model, topic_num: int):
        """This clears the chat history and starts again with a new news summary"""
        documents, document_scores, document_ids = topic_model.search_documents_by_topic(topic_num=topic_num, num_docs=10)
        newsrag = RAGSummariser()
        streamer = StreamingText()
        newsrag.llm.streaming_callback = streamer

        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(newsrag.run, kwds={"documents": [Document(content=d) for d in documents]}, error_callback=lambda x: print("Error in generation thread: ", x))

        history = [{"role": "assistant", "content": ""}]
        for new_token in iter(streamer):
            history[0]["content"] += new_token
            yield history
        
        pipeline_result = async_result.get()
        yield [{"role": "assistant", "content": pipeline_result["llm"]["replies"][0]}]

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
        history[-1] = {"role": "assistant", "content": pipeline_result["llm"]["replies"][0]}
        yield history, pipeline_result["prompt_builder"]["prompt"]

    #text.submit(summarise, [text], [chatbot])
    demo.load(topics, outputs=[topic_model, topic_selection])
    topic_selection.select(summarise, [topic_model, topic_selection], [chatbot])
demo.launch()