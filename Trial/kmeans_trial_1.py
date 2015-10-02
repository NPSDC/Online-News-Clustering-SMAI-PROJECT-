from sklearn.cluster import k_means
import pickle

filename = 'tfidf.pickle'
f = open(filename, 'rb')
data = pickle.load(f)
f.close()
ans = k_means(X = data,n_clusters =  20, max_iter = 100, n_init= 5, init = 'k-means++')
print ans[1]