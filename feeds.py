import feedparser
from haystack import Document
import arrow

from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html: str):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def get_date(date_string: str):
    """Return a date object from an RSS formatted date string"""
    arrow.get(date_string, "ddd, DD MMM YYYY HH:mm:ss ZZZ")


class Feed:
    """
    Base class for feeds specific to a vendor. This allows
    customisation of how inheriting feeds are processed
    """

    _url = None

    def feed2doc(self, item):
        return Document(content=strip_tags(item["summary"]),
                        meta={
                            "date": get_date(item["published"]),
                            "url": item["link"],
                            "title": item["title"]
                        })

    def get_documents(self):
        return [self.feed2doc(item) for item in self.parse()]

    def parse(self):
        d = feedparser.parse(self._url)
        return d["entries"]

class TheGuardian(Feed):
    _url = "https://theguardian.com/uk/rss"
     
    def feed2doc(self, item):
        content = strip_tags(item["summary"]).rstrip("Continue reading...")
        return Document(content=item["title"],
                        meta={
                            "date": get_date(item["published"]),
                            "url": item["link"],
                            "title": item["title"]
                        })


class BBC(Feed):
    _url = "https://feeds.bbci.co.uk/news/rss.xml"

    def feed2doc(self, item):
        """BBC summaries require the title for context"""
        content = strip_tags(item["summary"])

        return Document(content="\n".join([item["title"], content]),
                        meta={
                            "date": get_date(item["published"]),
                            "url": item["link"],
                            "title": item["title"]
                        })
    
class AssociatedPress(Feed):
    _url = "https://news.google.com/rss/search?q=when:24h+allinurl:apnews.com&hl=en-GB&gl=GB&ceid=GB:en"
    def feed2doc(self, item):
        """AP google news feeds have nothing in the summary"""
        title = item["title"].rstrip(" - The Associated Press")
        return Document(content=title,
                        meta={
                            "date": get_date(item["published"]),
                            "url": item["link"],
                            "title": title
                        })

def download_feeds(feed_cls: list):
    unique_docs = {}
    for feed in feed_cls:
        docs = feed().get_documents()
        for doc in docs:
            unique_docs[doc.id] = doc
    return list(unique_docs.values())
