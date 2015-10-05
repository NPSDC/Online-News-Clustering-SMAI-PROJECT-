import feedparser
from bs4 import BeautifulSoup
from urllib2 import urlopen

def parseRSS(rssurl):
	feed = feedparser.parse(rssurl)
	

def main():
	home = "https://news.google.com/news?cf=all&hl=en&pz=1&ned=in&output=rss"