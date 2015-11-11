import numpy as np
import dill as pickle

frozen = list()

class Cluster(object):
	""""""
	def __init__(self,  dim, cid, documents, doc_grp, frozen = 0,  length = 0):
		self.centroid = np.empty(dim)
		self.frozen = frozen
		self.documents = documents
		self.doc_grp = doc_grp
		self.length = length
		self.dim = dim
		self.id = cid

	def freeze(self):
		global frozen
		self.frozen = 1
		frozen.append(self)

	def compute_centroid(self, doc):
		self.centroid = (self.centroid*self.length + doc)/(self.length + 1)
		if (self.centroid == 0.).all():
			print self.id
		self.length += 1

	def add_document(self, doc, doc_num, doc_gr_id):
		self.documents[doc_num] = doc
		self.doc_grp[doc_gr_id] = doc_num
		self.compute_centroid(doc)

class Corpus(object):
	"""docstring for Corpus"""
	def __init__(self, X, max_n, doc_gr_id ,global_index = []):
		self.list_of_clusters = np.empty(max_n, dtype = Cluster)
		self.max_n = max_n #Max number of clusters
		self.labels = {}
		self.global_index = []
		self.doc_gr_id = doc_gr_id
		self.no_of_docs = 1
		self.X = X
		self.cid = 1

	def add_to_cluster(self, Inc_K_Means):
		for i in xrange(self.X.shape[0]):
			self.cid += Inc_K_Means.add_to_cluster(self.cid , self.X[i], \
				self.no_of_docs, self.doc_gr_id[i], self.list_of_clusters, self.labels, self.max_n)
			self.no_of_docs += 1
			

	def comp_labels(self, Inc_K_Means, length):
		Inc_K_Means.compute_labels(self.labels, self.list_of_clusters, length)


	def	do_stuff(self, Inc_K_Means, length):
		self.add_to_cluster(Inc_K_Means)

class Inc_K_Means(object):
	""" aa"""
	def __init__(self, threshold = 0.4):
		self.threshold = threshold

	def cos_similarity(self, cl1, cl2):
		# if (cl1 == 0).all():
		# 	print "1"
		# if (cl1 == 0).all():
		# 	print "2"
		return np.dot(cl1, cl2)/np.linalg.norm(cl1)/np.linalg.norm(cl2)

	def add_to_cluster(self, cid, doc, doc_num, doc_gr_id, list_of_clusters, labels, length):
		cluster_created = 0
		clust_id = 0
		for i in xrange(length):
			if list_of_clusters[i] is None:
				list_of_clusters[i] = Cluster(doc.shape[0], cid, dict(), dict())
				list_of_clusters[i].add_document(doc, doc_num, doc_gr_id)
				clust_id = list_of_clusters[i].id
				cluster_created = 1
				break
				
			else:				
				sim = self.cos_similarity(doc, list_of_clusters[i].centroid)
				document_keys = list_of_clusters[i].doc_grp.keys()
				flag = 0
				if(doc_gr_id in document_keys):
					flag = 1

				if(sim > self.threshold and flag == 0):
					list_of_clusters[i].add_document(doc, doc_num, doc_gr_id)	
					clust_id = list_of_clusters[i].id
					cluster_created = 0
					break
		else:
			new_clust = Cluster(doc.shape[0], cid, dict(), dict())
			new_clust.add_document(doc, doc_num, doc_gr_id)
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

def random():
	X = list()
	for j in xrange(10):
		for i in xrange(10):
			boolarr = (( np.random.rand(4) > 0.5 ) * 2) - 1
			X.append(np.random.rand(4) * boolarr )
	return np.array(X)

def truths(topics, X, doc_gr_id):
	dim = X.shape[1]
	truths = dict()
	truths2 = dict()
	clusters = list()
	for i in xrange(len(topics)):
		if not topics[i] in truths:
			truths[topics[i]] = list()
		if not topics[i] in truths2:
			truths2[topics[i]] = dict()
		if not doc_gr_id[i] in truths2[topics[i]]:
			truths2[topics[i]][doc_gr_id[i]] = list()
		truths2[topics[i]][doc_gr_id[i]].append(i+1)
		truths[topics[i]].append(i+1)
	#print truths2
	for topic in truths:
		document_ids = truths[topic]
		doc_grp_ids = truths2[topic]
		document_grp_ids = truths2[topic].keys()
		#print document_grp_ids
		document_dic = { doc: X[doc - 1] for doc in document_ids }
		doc_grp = { grp_id: truths2[topic][grp_id] for grp_id in document_grp_ids }
		cluster = Cluster( dim, topic, document_dic, doc_grp, True, len(document_ids) )
		cluster.centroid = X[np.array(document_ids) - 1 ].mean(axis=0)
		if (cluster.centroid == 0.).all():
			print cluster.id
		clusters.append(cluster)
	return np.array(clusters)

def main():
	X, topics, title, days, doc_gr_id = pickle.load(open('tfidf_fir_post.pickle', 'rb'))
	#print doc_gr_id
	data_cor = Corpus(max_n = 135, X = X, doc_gr_id = doc_gr_id)
	kmeans = Inc_K_Means(0.4)
	data_cor.do_stuff(kmeans, 135)
	# print data_cor.cid
	# print sorted(data_cor.list_of_clusters[2].documents.keys())
	# print data_cor.labels
	# print data_cor.cid
	# return (X,data_cor.cid)
	generated = frozen
	for cluster in data_cor.list_of_clusters:
		if cluster != None:
			generated.append(cluster)
	reference = truths(topics, X, doc_gr_id)
	pickle.dump({'reference': reference, 'generated': generated}, open("clusters_first_post.pickle", "wb"))
	
def sub():
	while(1):
		X, data_cor_cid = main()
		if( data_cor_cid > 50):
			pickle.dump(X, open('Y.pickle', 'wb'))
			print X, data_cor_cid
			break

if(__name__ == '__main__'):
	main()