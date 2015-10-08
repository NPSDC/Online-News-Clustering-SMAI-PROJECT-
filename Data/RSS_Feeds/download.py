import feedparser
from bs4 import BeautifulSoup
from urllib2 import urlopen

def getRSScoverage(rssurl):
	feed = feedparser.parse(rssurl)
	coverages = dict()
	for entry in feed['entries']:
		cluster_title = entry['title']
		title_article = entry['link']
		summary_links = BeautifulSoup(entry['summary'])
		cluster_links = [ tag_a.get("href") for tag_a in summary_links.findAll("a") ]
		coverage_page = urlopen(cluster_links[-1])
		coverages[cluster_title] = coverage_page
	return coverages

	

def main():
	home = "https://news.google.com/news?cf=all&hl=en&pz=1&ned=in&output=rss"

	# 'coverages' is a dict of coverage pages
	# the keys of each coverage page are the titles
	# each coverage pages has links to the actual articles
	coverages = getRSScoverage(home)
	# coverage_pages = [ coverage_page for coverage_page in coverages.values() ]
	# coverage_title = [ coverage_title for coverage_title in coverages.keys() ]