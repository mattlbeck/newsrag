from newsrag.feeds import BBC
import pytest
from collections import Counter

@pytest.mark.vcr()
def test_bbc_feeds():
    docs = BBC().get_documents()
    assert len(Counter([d.meta['subfeed'] for d in docs])) == len(BBC.subfeeds)
