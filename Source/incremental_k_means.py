import numpy as np
import pickle

class Cluster(object):
	""""""
	def __init__(self,  dim, cid, documents,frozen = 0,  length = 0):
		self.centroid = np.empty(dim)
		self.frozen = frozen
		self.documents = documents
		self.length = length
		self.dim = dim
		self.id = cid

	def freeze(self):
		self.frozen = 1

	def compute_centroid(self, doc):
		self.centroid = (self.centroid*self.length + doc)/(self.length + 1)
		self.length += 1

	def add_document(self, doc, doc_num):
		self.documents[doc_num] = doc
		self.compute_centroid(doc)

class Corpus(object):
	"""docstring for Corpus"""
	def __init__(self, X, max_n, global_index = []):
		self.list_of_clusters = np.empty(max_n, dtype = Cluster)
		self.max_n = max_n #Max number of clusters
		self.labels = {}
		self.global_index = []
		self.no_of_docs = 1
		self.X = X
		self.cid = 1

	def add_to_cluster(self, Inc_K_Means):
		for i in xrange(self.X.shape[0]):
			self.cid += Inc_K_Means.add_to_cluster(self.cid , self.X[i], \
				self.no_of_docs, self.list_of_clusters, self.labels, self.max_n)
			self.no_of_docs += 1
			

	def comp_labels(self, Inc_K_Means, length):
		Inc_K_Means.compute_labels(self.labels, self.list_of_clusters, length)


	def	do_stuff(self, Inc_K_Means, length):
		self.add_to_cluster(Inc_K_Means)
		#self.comp_labels(Inc_K_Means, length)

class Inc_K_Means(object):
	""" aa"""
	def __init__(self, threshold = 0.4):
		self.threshold = threshold

	def cos_similarity(self, cl1, cl2):
		return np.dot(cl1, cl2)/np.linalg.norm(cl1)/np.linalg.norm(cl2)

	def add_to_cluster(self, cid, doc, doc_num, list_of_clusters, labels, length):
		cluster_created = 0
		clust_id = 0
		for i in xrange(length):
			if list_of_clusters[i] is None:
			#	print i,"YES"
				list_of_clusters[i] = Cluster(doc.shape[0], cid, dict())
				list_of_clusters[i].add_document(doc, doc_num)
				clust_id = list_of_clusters[i].id
				cluster_created = 1
				break
				
			else:				
				sim = self.cos_similarity(doc, list_of_clusters[i].centroid)
				#print "UP\t",sim
				if(sim > self.threshold):
				#	print "Down\t",sim
					list_of_clusters[i].add_document(doc, doc_num)	
					clust_id = list_of_clusters[i].id
					cluster_created = 0
					break
		else:
			new_clust = Cluster(doc.shape[0], cid, dict())
			new_clust.add_document(doc, doc_num)
			clust_id = new_clust.id
			self.freeze_cluster(new_clust, list_of_clusters, length)
			cluster_created = 1
		self.compute_labels(labels, doc_num, clust_id)
		return cluster_created

	def freeze_cluster(self, new_clust, list_of_clusters, length):
		min_index_cluster = 0
		most_old = list_of_clusters[0].documents.keys()[-1]
		for i in xrange(length):
			if(list_of_clusters[i].documents.keys()[-1] < most_old ):
				most_old = list_of_clusters[i].documents.keys()[-1]
				min_index_cluster = i
		list_of_clusters[min_index_cluster].freeze()
		list_of_clusters[min_index_cluster] = new_clust

	def compute_labels(self, labels, doc_num, clust_id):
		labels[doc_num] = clust_id
		#print labels

def random():
	X = list()
	for j in xrange(10):
		for i in xrange(10):
			boolarr = (( np.random.rand(4) > 0.5 ) * 2) - 1
			X.append(np.random.rand(4) * boolarr )
	return np.array(X)

def main():
	# X = random()
	
	X = pickle.load(open('Y.pickle', 'rb'))
	#print X
	data_cor = Corpus(max_n = 20 , X = X)
	kmeans = Inc_K_Means(0.4)
	data_cor.do_stuff(kmeans, 20)
	#print sorted(data_cor.list_of_clusters[2].documents.keys())
	#print X
	#print "#############################"
	#print data_cor.labels
	#print data_cor.cid
	#eturn (X,data_cor.cid)
	'''for key in data_cor.labels:
		if data_cor.labels[key] == 3:
			print key'''
#cluster1 = Cluster(dim = len(data_cor.global_index))
	
def sub():
	while(1):
		X, data_cor_cid = main()
		if( data_cor_cid > 50):
			pickle.dump(X, open('Y.pickle', 'wb'))
			print X, data_cor_cid
			break

if(__name__ == '__main__'):
	main()