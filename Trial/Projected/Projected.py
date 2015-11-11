from sklearn.cluster import k_means

class Projected(object):
	""""""
	def __init__(self, dim, max_dim, no_of_clust, X):
		self.dim = dim
		self.max_dim = max_dim
		self.no_of_clust = no_of_clust
		##Not valid when number of data points less than number of clusters
		self.centroid, self.label, self.inertia = k_means(X, self.no_of_clust, init = 'k-means++')

