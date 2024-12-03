"""
Implements opinionated parsing of feeds into haystack Document objects.

This module also acts as a library of implemented feed parsers for different vendors. 
"""

from collections import defaultdict
import inspect
import re
import sys
from html.parser import HTMLParser
from io import StringIO

import arrow
import feedparser
from haystack import Document


def strip_tags(html: str):
    """Strip html tags from a string"""
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
                            "timestamp": get_date(item["published"]),
                            "published": item["published"],
                            "link": item["link"],
                            "title": item["title"],
                            "vendor": self.name
                        }
        default_meta.update(meta)
        return Document(content=content, meta=default_meta)
    
    def parse(self, item: dict) -> tuple[str, dict]:
        """
        Parse a dictionary for a feed entry to extract the content and metadata.

        This is an overridable method that must return a tuple of
        (content, meta: dict) given the item, which is the parsed
        entry from an RSS feed. The returned data is then added to the information used
        to construct a Document in `_feed2doc`
        """
        return item["title"], {}

    def get_documents(self) -> list[Document]:
        """
        Download and parse all documents for all subfeeds.

        If `self.subfeeds` is none, it will assume the only feed is
        defined by the base `self._url`. In that case the subfeed will
        be named "main".

        :returns: a flat list of Document objects for all subfeeds.
        """
        
        if not self.subfeeds:
            return self.get_subfeed()
        
        # Parse all subfeeds and deduplicate.
        # This uses the content as the ID for the document because the document
        # ID is affected by subfeed information.
        unique_docs = defaultdict(list)
        for name in self.subfeeds.keys():
            for doc in self.get_subfeed(name):
                unique_docs[doc.content].append(doc)
        
        deduplicated_documents = []
        # convert subfeed meta to list of subfeeds
        for same_docs in unique_docs.values():
            # use the 0th doc as the candidate and copy across the subfeed info from the others
            doc = same_docs[0]
            doc.meta["subfeeds"] = [d.meta["subfeeds"] for d in same_docs]
            deduplicated_documents.append(doc)
        return deduplicated_documents

        
    
    def get_subfeed(self, name=None) -> list[Document]:
        """
        Download and parse all entries of a subfeed.
        
        :param name: 
            The identifier of the subfeed. Must be in `self.subfeeds`. 
            If None, assumes there is only one subfeed given by the base URL
        :returns: a list of Document objects for each subfeed entry.
        """
        if name:
            try:
                modifier = self.subfeeds[name]
            except KeyError:
                raise ValueError(f"No such subfeed {name} in feed {self.name}")
        else:
            name = ["main"]
            modifier = ""

        url = self._url.format(modifier)
        feed = parse_feed(url)
        if len(feed) == 0:
            print(f"Warning: {url} has no entries.")
        docs = []
        for entry in feed:
            try:
                content, meta = self.parse(entry)
                docs.append(self._feed2doc(entry, content, subfeeds=name, **meta))
            except KeyError:
                print("Bad feed:", str(entry))
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
    _url = "https://abcnews.go.com/abcnews/{}"
    subfeeds = {
        "International Headlines": "internationalheadlines",
        "Top Stories": "topstories",
        "US Headlines": "usheadlines",
        "Politics Headlines": "politicsheadlines",
        "Business Headlines": "businessheadlines",
        "Technology Headlines": "technologyheadlines",
        "Health Headlines": "healthheadlines",
        "Entertainment Headlines": "entertainmentheadlines",
        "Travel Headlines": "travelheadlines",
        "ESPN Sports": "sportsheadlines",
    }

    def parse(self, item):
        return item["description"], {}
    
class FoxNews(Feed):
    name = "Fox News"
    _url = "https://moxie.foxnews.com/google-publisher/{}.xml"
    subfeeds = {
        "Latest Headlines": "latest",
        "World": "world",
        "Politics": "politics",
        "Science": "science",
        "Health": "health",
        "Sports": "sports",
        "Travel": "travel",
        "Tech": "tech",
        "Opinion": "opinion",
    }

    def parse(self, item):
        """Fox news has a 'content' field in which the whole article appears"""
        return item["description"], {}

class EuroNews(Feed):
    name = "Euro News"
    _url = "https://www.euronews.com/rss?name={}"
    subfeeds = {
        "Latest News": "news",
        "No Comment": "nocomment",
        "Voyage": "travel",
        "My Europe": "my-europe",
        "Sport": "sport",
        "Culture": "culture",
        "Next": "next",
        "Green": "green"
    }

    def parse(self, item):
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
    _url = "https://www.wired.com/feed/{}/rss"
    subfeeds = {
        "Business": "category/business/latest",
        "Artificial Intelligence": "tag/ai/latest",
        "Culture": "category/culture/latest",
        "Gear": "category/gear/latest",
        "Ideas": "category/ideas/latest",
        "Science": "category/science/latest",
        "Security": "category/security/latest",
        "Backchannel": "category/backchannel/latest"
    }
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
