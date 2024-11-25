import feedparser
from haystack import Document
import arrow
import re

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
    try:
        return arrow.get(date_string, "ddd, DD MMM YYYY HH:mm:ss ZZZ").timestamp()
    except arrow.parser.ParserMatchError:
        return arrow.get(date_string, "ddd, DD MMM YYYY HH:mm:ss Z").timestamp()


class Feed:
    """
    Base class for feeds specific to a vendor. This allows
    customisation of how inheriting feeds are processed
    """
    name = None
    _url = None

    def _feed2doc(self, item, content, **meta):
        default_meta = {
                            "date": get_date(item["published"]),
                            "link": item["link"],
                            "title": item["title"],
                            "vendor": self.name
                        }
        default_meta.update(meta)
        return Document(content=content, meta=default_meta)
    
    def feed2doc(self, item):
        return self._feed2doc(item, item["title"])

    def get_documents(self):
        return [self.feed2doc(item) for item in self.parse()]

    def parse(self):
        d = feedparser.parse(self._url)
        return d["entries"]
    
# Inheriting feeds are translated to documents with whatever preprocessing is needed
# Various fields are added to the main content depending on how the vendor uses the feed
class CNBC(Feed):
    name = "CNBC"
    _url = "https://www.cnbc.com/id/100727362/device/rss/rss.html"

class ABC(Feed):
    name = "ABC"
    _url = "https://abcnews.go.com/abcnews/internationalheadlines"

    def feed2doc(self, item):
        return self._feed2doc(item, item["description"])
    
class FoxNews(Feed):
    name = "Fox News"
    _url = "https://moxie.foxnews.com/google-publisher/latest.xml"

    def feed2doc(self, item):
        """Fox news has a 'content' field in which the whole article appears"""
        return self._feed2doc(item, item["description"])

class TheCipherBrief(Feed):
    """This may not be an ideal feed - description is much of the article and titles are not very headline-y"""
    _url = "https://www.thecipherbrief.com/feed"

class EuroNews(Feed):
    name = "Euro News"
    _url = "https://www.euronews.com/rss"

    def feed2doc(self, item):
        """Fox news has a 'content' field in which the whole article appears"""
        return self._feed2doc(item, item["description"])
    
class TheGuardian(Feed):
    name = "The Guardian"
    _url = "https://theguardian.com/uk/rss"
     
    def feed2doc(self, item):
        content = strip_tags(item["summary"]).rstrip("Continue reading...")
        item["title"] = re.sub(r"\|(.+)$", "", item["title"]) # remove author attributions from the end of titles
        return self._feed2doc(item, item["title"])


class BBC(Feed):
    name = "BBC"
    _url = "https://feeds.bbci.co.uk/news/rss.xml"

    def feed2doc(self, item):
        """BBC summaries require the title for context"""
        content = strip_tags(item["summary"])
        return self._feed2doc(item, "\n".join([item["title"], content]))
    
class AssociatedPress(Feed):
    name = "The Associated Press"
    _url = "https://news.google.com/rss/search?q=when:24h+allinurl:apnews.com&hl=en-GB&gl=GB&ceid=GB:en"
    def feed2doc(self, item):
        """AP google news feeds have nothing in the summary"""
        title = item["title"].rstrip(" - The Associated Press")
        return self._feed2doc(item, title, title=title)
    

# tech
class TechCrunch(Feed):
    name = "TechCrunch"
    _url = "https://techcrunch.com/feed/"
    def feed2doc(self, item):
        content = strip_tags(item["description"]).rstrip("© 2024 TechCrunch. All rights reserved. For personal use only.")
        content = item["title"] + " " + content
        return self._feed2doc(item, content)
    
class Wired(Feed):
    name = "Wired"
    _url = "https://www.wired.com/feed/rss"
    def feed2doc(self, item):
        return self._feed2doc(item, item["description"])
    
class ArsTechnica(Feed):
    name = "Ars Technica"
    _url = "https://feeds.arstechnica.com/arstechnica/technology-lab"
    def feed2doc(self, item):
        content = strip_tags(item["description"])
        content = item["title"] + " " + content
        return self._feed2doc(item, content)

# Science
class NewScientist(Feed):
    name = "New Scientist"
    _url = "https://www.newscientist.com/section/news/feed/"
    def feed2doc(self, item):
        content = strip_tags(item["description"])
        content = item["title"] + " " + content
        return self._feed2doc(item, content)

def download_feeds(feed_cls: list):
    unique_docs = {}
    for feed in feed_cls:
        docs = feed().get_documents()
        for doc in docs:
            unique_docs[doc.id] = doc
    return list(unique_docs.values())
