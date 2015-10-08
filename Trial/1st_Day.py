import urllib2
from bs4 import BeautifulSoup
BS = BeautifulSoup

def set_url_list(url, main_url, topics_wise):
	#print url
	f = urllib2.urlopen(url)
	data = f.read()
	f.close()
	soup = BS(data)
	sel_urls = {}
	sel_url_div = soup.findAll('div', {'class':'topic'})
	if sel_url_div :
		for divs in sel_url_div:
			a_tag = divs.findAll('a', {'class':'persistentblue'})[0]
			text = a_tag.text
			href = main_url + a_tag.get('href')
			sel_urls[text] = href
			topics_wise[soup.findAll('span', {'class':'sel'})[0].text] = sel_urls
	return topics_wise, soup

def set_main_topics(main_html, main_url, topics_wise):
	side_bar = main_html.findAll('div', {'class':'browse-sidebar'})
	other_topics = side_bar[0].findAll('li')[1:]
	for topic in other_topics:
		a_tag = topic.findAll('a', {'class':'persistentblue'})[0]
		text = a_tag.text
		href = main_url + a_tag.get('href')
		topics_wise[text] = href
	return topics_wise

def main():

	main_url = 'https://news.google.com'
	topics_wise = {}
	topics_wise, soup = set_url_list(main_url, main_url, topics_wise)
	topic_1 = topics_wise.keys()[0]
	topics_wise = set_main_topics(soup, main_url, topics_wise)
#	print topics_wise
	for topic in topics_wise.keys():
		if(topic != topic_1):
#			print topics_wise[topic]
			topics_wise, soup = set_url_list(topics_wise[topic], main_url, topics_wise)

	print topics_wise

if __name__ == '__main__':
	main()