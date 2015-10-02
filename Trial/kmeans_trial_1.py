from sklearn.cluster import k_means
import pickle

filename = 'tfidf.pickle'
f = open(filename, 'rb')
data = pickle.load(f)
f.close()

tfidf, titles = data
ans = k_means(X = tfidf, n_clusters =  20, max_iter = 100, n_init= 5, init = 'k-means++')

results = dict()
for i in xrange(len(ans[1])):
	c_i = ans[1][i]
	if not c_i in results.keys():
		results[c_i] = list()
	results[c_i].append(titles[i])

for key in results:
	print key
	print "="*10
	for title in results[key]:
		print title
	print