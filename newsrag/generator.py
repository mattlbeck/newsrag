import re
from queue import Queue

from haystack import Document


def transform_citations(citation: str,) -> list[int]:
    cite_numbers = []
    m = re.match(r"(.*\[)(.+)(\].*)", citation, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Bad citation format: {citation}")
    start, content, end = m.groups()
    m = re.findall(r"(?:ARTICLE\s(\d+))+", content)
    if not m:
        return start, [], end
    else:
        for cite in m:
            cite_numbers.append(int(cite))
    return start, cite_numbers, end


class StreamingText:
    """A callback that that accepts streaming output from a model"""
    def __init__(self):
        self._text = Queue()
        self._done = False

    def __call__(self, text_chunk):
        
        # stop code from ollama
        if "done" in text_chunk.meta and text_chunk.meta["done"]:
            self._done = True

        # stop code from huggingface API
        if "finish_reason" in text_chunk.meta:
            self._done = True
        self._text.put(text_chunk.content)

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration()
        return self._text.get()


class Sources:
    """Manages a list of sources that can be generated as a bibliography"""
    def __init__(self):

        self._ids = []
        self._sources = []

    def add_source(self, document: Document) -> int:
        """Add a new source if it ist not already in the source list.
        
        Args:
            document: the document to add as a source
        
        returns: The unique citation number of the document in the source list.
        """
        try:
            return self._ids.index(document.id) + 1
        except ValueError:
            self._ids.append(document.id)
            self._sources.append(document)
            return len(self._sources)

    def generate_bibliography(self):
        """Generates formatted strings representing each source."""
        for i, source in enumerate(self._sources):
            title = source.meta["title"]
            link = source.meta["link"]
            vendor = source.meta["vendor"]
            yield f"{i+1}. {title} - [{vendor}]({link})"


def stream_sourced_output(stream, sources: Sources, documents: list[Document]) -> tuple[list, Sources]:
    history = ""
    
    ref = ""
    for new_token in stream:
        if "[" in new_token or ref:
            ref += new_token
        if ref and "]" in new_token:
            start, citations, end = transform_citations(ref)
            if not citations:
                new_token = ["BAD REF"]

            new_citations = []
            for cite in citations:
                doc = documents[cite - 1]
                new_ref = sources.add_source(doc)
                new_citations.append(new_ref)
            new_token = (start + ','.join(str(cite) for cite in new_citations) + end)
            ref = ""
            
        if ref:
            continue
        history += new_token
        yield history, sources
