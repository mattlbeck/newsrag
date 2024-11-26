import inspect
import re
import sys
from html.parser import HTMLParser
from io import StringIO

import arrow
import feedparser
from haystack import Document


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

def parse_feed(url):
    d = feedparser.parse(url)
    return d["entries"]

class Feed:
    """
    Base class for feeds specific to a vendor. This allows
    customisation of how inheriting feeds are processed
    """
    name = None
    _url = None
    subfeeds = None

    def _feed2doc(self, item, content, **meta):
        default_meta = {
                            "date": get_date(item["published"]),
                            "link": item["link"],
                            "title": item["title"],
                            "vendor": self.name
                        }
        default_meta.update(meta)
        return Document(content=content, meta=default_meta)
    
    def parse(self, item):
        """
        overridable method that must return a tuple of
        (content, meta: dict) given the item, which is the parsed
        entry from an RSS feed.
        """
        return item["title"], {}

    def get_documents(self):
        docs = []
        if self.subfeeds:
            for name, modifier in self.subfeeds.items():
                url = self._url.format(modifier)
                feed = parse_feed(url)
                for entry in feed:
                    content, meta = self.parse(entry)
                    docs.append(self._feed2doc(entry, content, subfeed=name, feed_url=url, **meta))
        else:
            feed = parse_feed(self._url)
            for entry in feed:
                content, meta = self.parse(entry)
                docs.append(self._feed2doc(entry, content, **meta))
        return docs

        
    
# Inheriting feeds are translated to documents with whatever preprocessing is needed
# Various fields are added to the main content depending on how the vendor uses the feed
class CNBC(Feed):
    name = "CNBC"
    _url = "https://www.cnbc.com/id/100727362/device/rss/rss.html"
    subfeeds = {
        "Top News": "100727362",
        "U.S. News": "15837362",
        "Asia News": "19832390",
        "Europe News": "19794221",
        "Business News": "10001147",
        "Technology": "19854910",
        "Health Care": "10000108"

    }

class ABC(Feed):
    name = "ABC"
    _url = "https://abcnews.go.com/abcnews/internationalheadlines"

    def parse(self, item):
        return item["description"], {}
    
class FoxNews(Feed):
    name = "Fox News"
    _url = "https://moxie.foxnews.com/google-publisher/latest.xml"

    def parse(self, item):
        """Fox news has a 'content' field in which the whole article appears"""
        return item["description"], {}

class TheCipherBrief(Feed):
    """This may not be an ideal feed - description is much of the article and titles are not very headline-y"""
    _url = "https://www.thecipherbrief.com/feed"

class EuroNews(Feed):
    name = "Euro News"
    _url = "https://www.euronews.com/rss"

    def parse(self, item):
        """Fox news has a 'content' field in which the whole article appears"""
        return item["description"], {}
    
class TheGuardian(Feed):
    name = "The Guardian"
    _url = "https://theguardian.com/{}/rss"
    subfeeds = {
        "UK": "uk",
        "World": "world",
        "Business": "business",
        "Environment": "environment",
        "UK Politics": "politics",
        "Tech": "technology",
        "Society": "society",
        "US Politics": "us-news/us-politics"
    }
      
    def parse(self, item):
        content = strip_tags(item["summary"]).rstrip("Continue reading...")
        item["title"] = re.sub(r"\|(.+)$", "", item["title"]) # remove author attributions from the end of titles
        return item["title"], {}


class BBC(Feed):
    name = "BBC"
    _url = "https://feeds.bbci.co.uk/news{}/rss.xml"
    subfeeds = {    
        "Top Stories": "",
        "World": "/world",
        "UK": "/uk",
        "Business": "/business",
        "Politics": "/politics",
        "Health": "/health",
        "Education & Family": "/education",
        "Science & Environment": "/science_and_environment",
        "Technology": "/technology",
        "Entertainment and Arts": "/entertainment_and_arts"
    }

    def parse(self, item):
        """BBC summaries require the title for context"""
        content = strip_tags(item["summary"])
        return "\n".join([item["title"], content]), {}
    
class AssociatedPress(Feed):
    name = "The Associated Press"
    _url = "https://news.google.com/rss/search?q=when:24h+allinurl:apnews.com&hl=en-GB&gl=GB&ceid=GB:en"
    def parse(self, item):
        """AP google news feeds have nothing in the summary"""
        title = item["title"].rstrip(" - The Associated Press")
        return title, {"title": title}
    

# tech
class TechCrunch(Feed):
    name = "TechCrunch"
    _url = "https://techcrunch.com/feed/"
    def parse(self, item):
        content = strip_tags(item["description"]).rstrip("© 2024 TechCrunch. All rights reserved. For personal use only.")
        content = item["title"] + " " + content
        return content, {}
    
class Wired(Feed):
    name = "Wired"
    _url = "https://www.wired.com/feed/rss"
    def parse(self, item):
        return item["description"], {}
    
class ArsTechnica(Feed):
    name = "Ars Technica"
    _url = "https://feeds.arstechnica.com/arstechnica/technology-lab"
    def parse(self, item):
        content = strip_tags(item["description"])
        content = item["title"] + " " + content
        return content, {}

# Science
class NewScientist(Feed):
    name = "New Scientist"
    _url = "https://www.newscientist.com/section/news/feed/"
    def parse(self, item):
        content = strip_tags(item["description"])
        content = item["title"] + " " + content
        return content, {}


def download_feeds(feed_cls: list=None):
    if not feed_cls:
        # compile a list of all feeds defined in the module
        feed_cls = []
        for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            try:
                if obj == Feed:
                    continue
                if issubclass(obj, sys.modules[__name__].Feed):
                    feed_cls.append(obj)
            except AttributeError as e: 
                continue
        
    unique_docs = {}
    for feed in feed_cls:
        docs = feed().get_documents()
        for doc in docs:
            unique_docs[doc.id] = doc
    return list(unique_docs.values())
