import feedparser
from bs4 import BeautifulSoup
from urllib2 import urlopen
from collections import defaultdict

def getRSSlatest(rssurl):
	feed = feedparser.parse(rssurl)
	coverages = dict()
	for entry in feed['entries']:
		brief = BeautifulSoup(entry['summary'])
		links = [ tag_a.get("href") for tag_a in brief.findAll("a") ]
		event = links.pop()
		coverages[event] = links
	return coverages

def main():
	home = "https://news.google.com/news?cf=all&hl=en&pz=1&ned=in&output=rss"
	new_news = getRSSlatest(home)
	old_news = pickle.load(open('coverages.pickle', 'rb'))
	breaking = list()
	clusters = pickle.load(open('topicwise.pickle', 'rb'))
	for event in new_news:
		if event not in old_news:
			breaking.append(event)
		topic = old_news[event]
		if topic not in clusters:
			new_topic = defaultdict()
			new_topic.default_factory = new_topic.__len__
			clusters[topic] = new_topic
		for url in new_news[event]:
			if not url in clusters[topic]:
				articleID = clusters[topic][url]
				html_page = urlopen(url)
	# coverage_pages = [ coverage_page for coverage_page in coverages.values() ]
	# coverage_title = [ coverage_title for coverage_title in coverages.keys() ]
main()