import re
from queue import Queue
from typing import Generator

from haystack import Document


def transform_citations(citation: str) -> list[int]:
    """
    Extracts citations from the citation pattern [ARTICLE x, ARTICLE y].
    
    Arguments: citation (st)
    """
    cite_numbers = []
    # match the general [ ... ] pattern, capturing start, content and end
    m = re.match(r"(.*\[)(.+)(\].*)", citation, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Bad citation format: {citation}")
    start, content, end = m.groups()

    # extract article numbers
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


def stream_sourced_output(stream, sources: Sources, documents: list[Document]) -> Generator[tuple[list, Sources], None, None]:
    """Stream output that may contain citations that need to be parsed in stream.
    
    :param stream: a generator that will yield new tokens.
    :param sources: a running list of sources to add to.
    :param documents: the documents that may be sourced in the text stream.

    :yield: a tuple of the current output up till now, and the current source list.
    """
    history = ""
    
    ref = ""
    for new_token in stream:
        # cache tokens when a citation opener is found. The whole citation is then 
        # yielded only when it is complete and parsed.
        if "[" in new_token or ref:
            ref += new_token
        if ref and "]" in new_token:
            # if there is an ongoing citation and it is closed, parse these citations
            start, citations, end = transform_citations(ref)
            if not citations:
                new_token = ["BAD REF"]

            # for each citation found, retrieve the document it is citing and add it 
            # to the source list
            new_citations = []
            for cite in citations:
                doc = documents[cite - 1]
                # the ref may be different to the input ref if the document was already
                # in the source list.
                new_ref = sources.add_source(doc)
                new_citations.append(new_ref)

            # recompile the citation back into a string as the next token, including
            # the new reference ids that may have been referencing previous sources.
            new_token = (start + ','.join(str(cite) for cite in new_citations) + end)
            ref = ""
            
        if ref:
            continue
        history += new_token
        yield history, sources
