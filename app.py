import gradio as gr
from generator import Sources, StreamingText
from rag import RAGSummariser, get_document_store, JointDocumentIndexingPipeline, TopicModelPipeline, DescribeTopicPipeline, TopicRetrievalPipeline, QAGeneratorPipeline, QARetrievalPipeline
import feeds
import arrow

def get_bibliography(sources: Sources):
    source_list = []
    for i, source in enumerate(sources._sources):
        source_list.append(f"{i+1}. {source.meta['title']} - [{source.meta['vendor']}]({source.meta['link']})")
    return "\n".join(source_list)

def get_bibliography_table(sources: Sources):
    rows = []
    for i, source in enumerate(sources._sources):
        row = [str(i+1), source.meta["title"], source.meta["vendor"], f"[view source]({source.meta['link']})"]
        row_string = "|".join(row)
        rows.append("|"+row_string+"|")

    return "\n|-|-|-|-|\n".join(rows)


with gr.Blocks() as demo:

    def topics(document_store):
        """Downloads news feeds and indexes their content in to the central document store"""
        # download news from various feeds, formatted as haystack Document objects complete with some metadata
        news = feeds.download_feeds((feeds.TheGuardian, feeds.AssociatedPress, feeds.BBC, feeds.ABC, feeds.CNBC, feeds.FoxNews, feeds.EuroNews, feeds.TechCrunch, feeds.Wired, feeds.ArsTechnica))
        print(f"{len(news)} news articles")

        # Use the indexing pipeline to embed and write these documents to the chosen document store
        indexing = JointDocumentIndexingPipeline(document_store=document_store, min_word_count=0)
        indexing.run(news)

        # the topic pipeline discovers topics within the embedded documents and labels them with the embedded word vocabulary
        topics = TopicModelPipeline(document_store=document_store)
        result = topics.run(min_date=arrow.utcnow().shift(days=-1))

        # Describe each topic with a human readable title
        topic_describer = DescribeTopicPipeline()
        topic_descriptions = [topic_describer.run(topic) for topic in result["topic_model"]["topic_words"]]
        
        return gr.update(choices=topic_descriptions, value=None)
    
    def summarise(document_store, sources, topic_num: int):
        """Summarises the given topic by retrieving documents related to that topic and 
        putting them throuth the summariser pipeline.
        
        This clears the chat history and starts again with a new news summary"""
        documents = TopicRetrievalPipeline(document_store=document_store, document_count=30).run(topic_id=topic_num)
        newsrag = RAGSummariser()

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

    def user(user_message, history:list):    
        if history is None:
            history = []
        return history + [{"role": "user", "content": user_message}]
    
    def qa(document_store, sources, history: list):
        retriever = QARetrievalPipeline(document_store=document_store)
        
        # TODO: additional conversation context
        question = history[-1]["content"]
        documents = retriever.run(question)
        print("retrieved", documents, "documents")
        qa = QAGeneratorPipeline()
        
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
    sources = gr.State(value=Sources())
    document_store = gr.State(value=get_document_store())
    # arrange UI elements
    with gr.Row():
        title = gr.Markdown("# newsrag")
    with gr.Row():
        with gr.Column():
            topic_selection = gr.Dropdown(label="Select a topic", type="index")
            chatbot = gr.Chatbot(type="messages", height=600)
            qa_input = gr.Textbox(label="Ask a Question:")
        with gr.Column():
            bibliography = gr.Markdown(label="Bibliography", container=True, height=800)
    
    # set actions and triggers
    demo.load(topics, inputs=[document_store], outputs=[topic_selection])
    topic_selection.select(summarise, inputs=[document_store, sources, topic_selection], outputs=[chatbot, bibliography])
    qa_input.submit(user, inputs=[qa_input, chatbot], outputs=[chatbot]).then(qa, inputs=[document_store, sources, chatbot], outputs=[chatbot, bibliography])
demo.launch()
