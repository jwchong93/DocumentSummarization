import urllib3
from bs4 import BeautifulSoup

from News_Query.FrequencySummarizer import FrequencySummarizer


def get_only_text(url):
 """
  return the title and the text of the article
  at the specified url
 """
 page = urllib3.PoolManager().request('GET',url).data.decode('utf8')
 soup = BeautifulSoup(page,"html5lib")
 text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
 return soup.title.text, text

def get_document_and_summary_with_classical_method():
    feed_xml = urllib3.PoolManager().request('GET','http://feeds.bbci.co.uk/news/rss.xml').data
    feed = BeautifulSoup(feed_xml.decode('utf8'),"html5lib")
    to_summarize = list(map(lambda p: p.text, feed.find_all('guid')))

    fs = FrequencySummarizer()
    for article_url in to_summarize[:5]:
        title, text = get_only_text(article_url)
    #print (title)
    for s in fs.summarize(text, 2):
        yield text , s