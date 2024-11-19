from queue import Queue
import re
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


class Sources:

    def __init__(self):

        self._ids = []
        self._sources = []

    def add_source(self, document: Document) -> int:
        try:
            return self._ids.index(document.id) + 1
        except ValueError:
            self._ids.append(document.id)
            self._sources.append(document)
            return len(self._sources)

    def generate_bibliography(self):
        for i, source in enumerate(self._sources):
            title = source.meta["title"]
            link = source.meta["link"]
            vendor = source.meta["vendor"]
            yield f"{i+1}. {title} - [{vendor}]({link})"


def stream_sourced_output(stream, documents):
    history = ""
    sources = Sources()
    
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
