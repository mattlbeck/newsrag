import generator
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

def test_extract_citations():
    assert generator.transform_citations("[ARTICLE 1]") == ("[", [1], "]")
    assert generator.transform_citations("[ARTICLE 1, ARTICLE 2]") == ("[", [1,2], "]")
    assert generator.transform_citations(" [ARTICLE 1].\n") == (" [", [1], "].\n")


def test_stream_with_sources():

    text = "this is a statement [ARTICLE 1].\n This is another statement [ARTICLE 1, ARTICLE 11]"

    documents = [
        Document(content=f"This is article {i}") for i in range(13)
    ]
    tokens = []
    for i in range(0, len(text), 2):
        tokens.append(text[i:i+2])

    for content, sources in generator.stream_sourced_output(tokens, documents):
        continue

    assert content == "this is a statement [1].\n This is another statement [1,2]"
    assert sources._sources == documents


def test_joint_embedding():
    store = InMemoryDocumentStore()
    from rag import JointDocumentIndexingPipeline, get_document_store
    p = JointDocumentIndexingPipeline(store, min_word_count=0)
    p.run([Document(content="hello world")])
    assert store.count_documents() == 3
    