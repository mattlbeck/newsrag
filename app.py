import os
from collections import Counter
from pathlib import Path
from random import shuffle

import arrow
import gradio as gr
import jsonlines
from haystack import Document

import newsrag.feeds as feeds
from newsrag.config import AppConfig
from newsrag.generator import Sources
from newsrag.pipelines import (DescribeTopicPipeline,
                               JointDocumentIndexingPipeline,
                               QAGeneratorPipeline, QARetrievalPipeline,
                               SummarisationPipeline, TopicModelPipeline,
                               TopicRetrievalPipeline)
import yaml

DEFAULT_PARAMS = yaml.safe_load(open("params.yaml"))

description = """
# newsrag
**Summarisation of daily news topics**.
 - top2vec discovers topics from the last day's news headlines across a diverse number of news vendors
 - An LLM summarises each topic, building a list of citations as the response is streamed
 - Q&A allows you to ask questions against the database of news headlines in a classic RAG paradigm
"""

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
    env_var = "APP_DOCUMENT_CACHE"
    if env_var not in os.environ:
        return None
    cache_file = Path(os.environ["APP_DOCUMENT_CACHE"])
    if cache_file.exists():
        print(f"Loading news from cache {cache_file}")
        news = []
        with jsonlines.open(cache_file) as reader:
            for doc in reader:
                news.append(Document.from_dict(doc))
        return news


with gr.Blocks() as demo:
    config = AppConfig()

    def get_topics(document_store, min_date, n_neighbors, min_cluster_size, progress=gr.Progress()):
        """Downloads news feeds and indexes their content in to the central document store"""
        # download news from various feeds, formatted as haystack Document objects complete with some metadata
        progress(0.25, desc="Downloading news")
        news = get_cached_news()
        if not news: 
            print("Downloading fresh news")
            news = feeds.download_feeds()
            print(f"{len(news)} news articles")
        
        in_date_news = [doc for doc in news if doc.meta["timestamp"] >= min_date.timestamp()]
        print(f"{len(in_date_news)} news articles after filtering by date")

        progress(0.5, desc="Indexing news")
        # Use the indexing pipeline to embed and write these documents to the chosen document store
        indexing = JointDocumentIndexingPipeline(document_store=document_store, joint_embedder=config.get_joint_document_embedder(min_word_count=3))
        indexing.run(in_date_news)

        return model_topics(document_store, min_date, n_neighbors, min_cluster_size, progress=progress)
    
    def model_topics(document_store, min_date, n_neighbors, min_cluster_size, progress=gr.Progress()):
        """Models the topics, assuming they have already been indexed in the store"""
        # the topic pipeline discovers topics within the embedded documents and labels them with the embedded word vocabulary
        progress(0.75, desc="Discovering topics")
        topics = TopicModelPipeline(
            document_store=document_store, 
            umap_args={"n_neighbors": n_neighbors},
            hdbscan_args={"min_cluster_size": min_cluster_size}
        )
        result = topics.run(min_date=min_date)

        # Describe each topic with a human readable title
        progress(0.9, desc="Describing topics")
        topic_describer = DescribeTopicPipeline(generator=config.get_generator_model())
        topic_descriptions = [topic_describer.run(topic) for topic in result["topic_model"]["topic_words"]]

        # add a hint to the user for the size of each topic
        documents = result["topic_model"]["documents"]
        topic_outliers = [d for d in documents if d.meta["topic_outlier"]]
        topic_sizes = Counter([doc.meta["topic_id"] for doc in documents if not doc.meta["topic_outlier"]])
        topic_descriptions = [f"{description} ({topic_sizes[i]})" for i, description in enumerate(topic_descriptions)]
        topic_descriptions += [f"In other news... ({len(topic_outliers)})"]

        
        return gr.update(choices=topic_descriptions, value=None), result["topic_model"]["topic_words"]
    
    def summarise(document_store, sources, topics, topic_num: int):
        """Summarises the given topic by retrieving documents related to that topic and 
        putting them throuth the summariser pipeline.
        
        This clears the chat history and starts again with a new news summary"""
        print(f"number of topics: {len(topics)}, selected: {topic_num}")
        if topic_num >= len(topics):
            # retrieve miscelaneous news
            outlier_docs = document_store.filter_documents( {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.type", "operator": "==", "value": "document"},
                    {"field": "meta.topic_outlier", "operator": "==", "value": True}
                ]
            }) 
            shuffle(outlier_docs)
            documents = outlier_docs[:30]
        else:
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
    topics = gr.State(value=None)

    # arrange UI elements
    with gr.Row():
        with gr.Column(visible=False, min_width=200, scale=0) as sidebar:
            
            gr.DateTime.time_format = "%Y-%m-%d" # workaround for a gradio bug
            min_date = gr.DateTime(arrow.utcnow().shift(days=-1).datetime, type="datetime", include_time=False, label="Oldest news:")

            n_neighbors = gr.Slider(
                label="UMAP nearest neighbors", 
                minimum=1, 
                maximum=50, 
                value=DEFAULT_PARAMS["model_topics"]["umap"]["n_neighbors"]
            )
            min_cluster_size = gr.Slider(
                label="hdbscan minimum cluster size",
                minimum=2,
                maximum=30,
                value=DEFAULT_PARAMS["model_topics"]["hdbscan"]["min_cluster_size"]
            )
            refresh_topics = gr.Button(value="Refresh topics")


        with gr.Column():
            title = gr.Markdown(description)
            open_sidebar_btn = gr.Button("Show more options", scale=0)
            close_sidebar_btn = gr.Button("Hide more options", visible=False, scale=0)
            with gr.Tab("Select topic"):
                topic_selection = gr.Dropdown(label="", type="index")
            with gr.Tab("Ask a question"):
                qa_input = gr.Textbox(label="")



            bibliography = gr.Markdown(label="Bibliography", container=True, height=400)
            
        with gr.Column():
            chatbot = gr.Chatbot(type="messages", height=800)

            
    
    # sidebar show/hide logic
    open_sidebar_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    ), outputs=(open_sidebar_btn, close_sidebar_btn, sidebar))
    close_sidebar_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    ), outputs=(open_sidebar_btn, close_sidebar_btn, sidebar))
    
    # set actions and triggers
    get_topics_inputs = [document_store, min_date, n_neighbors, min_cluster_size]
    demo.load(get_topics, inputs=get_topics_inputs, outputs=[topic_selection, topics])
    refresh_topics.click(model_topics, inputs=[document_store, min_date], outputs=[topic_selection, topics])
    topic_selection.select(summarise, inputs=[document_store, sources, topics, topic_selection], outputs=[chatbot, bibliography])

    qa_input.submit(user_query, inputs=[qa_input, chatbot], outputs=[chatbot, qa_input]).then(qa, inputs=[document_store, sources, chatbot], outputs=[chatbot, bibliography])
demo.launch()
