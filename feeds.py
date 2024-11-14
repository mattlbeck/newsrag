import feedparser
from haystack import Document
import arrow


class Feed:
    """
    Base class for feeds specific to a vendor. This allows
    customisation of how inheriting feeds are processed
    """

    _url = None

    def feed2doc(self, item):
        return Document(content=item["summary"],
                        meta={
                            "date": arrow.get(item["published"], "ddd, DD MMM YYYY HH:mm:ss ZZZ")
                        })

    def get_documents(self):
        entries = self.parse()
        for item in entries:
            yield self.feed2doc(item)

    def parse(self):
        d = feedparser.parse(self._url)
        return d["entries"]

class TheGuardian(Feed):
    _url = "https://theguardian.com/uk/rss"