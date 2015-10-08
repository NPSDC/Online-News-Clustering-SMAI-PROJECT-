import urllib2
from bs4 import BeautifulSoup
BS = BeautifulSoup

main_url = 'https://news.google.com'
topics_wise = {}
#def set_url_list(url, main_url, topics_wise):
f = urllib2.urlopen('https://news.google.com/')
data = f.read()
f.close()
soup = BS(data)
sel_urls = {}

#def set_main_topics(main_html, ):

sel_url_div = soup.findAll('div', {'class':'topic'})
side_bar = soup.findAll('div', {'class':'browse-sidebar'})

for divs in sel_url_div:
	a_tag = divs.findAll('a', {'class':'persistentblue'})[0]
	text = a_tag.text
	href = main_url + a_tag.get('href')
	sel_urls[text] = href

topics_wise[soup.findAll('span', {'class':'sel'})[0].text] = sel_urls


other_topics = side_bar[0].findAll('li')[1:]
for topic in other_topics:
	a_tag = topic.findAll('a', {'class':'persistentblue'})[0]
	text = a_tag.text
	href = main_url + a_tag.get('href')
	topics_wise[text] = href
print topics_wise