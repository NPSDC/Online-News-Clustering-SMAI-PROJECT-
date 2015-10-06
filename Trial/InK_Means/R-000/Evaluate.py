import pickle
import numpy as np
from incremental_k_means import Cluster

def cos_similarity(cl1, cl2):
	return np.dot(cl1, cl2)/np.linalg.norm(cl1)/np.linalg.norm(cl2)

def compute_best(actual_clusters, gen_clusters, l_ac, l_gen):
	matched_clusters = np.empty(l_gen, dtype = int)
	for i in xrange(l_gen):
		max_sim = cos_similarity(actual_clusters[0].centroid, gen_clusters[i].centroid)
		max_index = 0
		for j in xrange(l_ac):
			sim = cos_similarity(actual_clusters[j].centroid, gen_clusters[i].centroid)
			if (sim > max_sim):
				max_sim = sim
				max_index = j
		matched_clusters[i] = max_index
		print max_sim, "\t", set(actual_clusters[matched_clusters[i]].documents.keys()).intersection(set(gen_clusters[i].documents.keys()))
	return matched_clusters

def compute_f_measure(actual_clusters, gen_clusters, l_ac, l_gen):
	matched_clusters = compute_best(actual_clusters, gen_clusters, l_ac, l_gen)
	precision = np.empty(l_gen, dtype = float)
	recall = np.empty(l_gen, dtype = float)
	f_measure = np.empty(l_gen, dtype = float)
	for i in xrange(l_gen):
		docs_gen = gen_clusters[i].documents.keys()
		docs_act = actual_clusters[matched_clusters[i]].documents.keys()
		common = len(set(docs_gen).intersection(set(docs_act)))
		print "actual\t", sorted(docs_act)
		print "found\t", sorted(docs_gen)
		if common == 0:
			f_measure[i] = 0.
			precision[i] = 0.
			recall[i]    = 0.
		if True:
			precision[i] = float(common)/gen_clusters[i].length
			recall[i] = float(common)/actual_clusters[matched_clusters[i]].length
			f_measure[i] = 2*precision[i]*recall[i]/(precision[i] + recall[i])
	return precision,recall, f_measure

def main():
	clusters = pickle.load(open("clusters.pickle", "rb"))
	actual_clusters = clusters['reference']; l_ac = len(actual_clusters)
	gen_clusters    = clusters['generated']; l_gen = len(gen_clusters)

	# for cluster in gen_clusters:
	# 	cluster.centroid = np.array(cluster.documents.values()).mean(axis=0)

	pre, rec, f_measure = compute_f_measure(actual_clusters, gen_clusters, l_ac, l_gen)
	# print f_measure

if __name__ == "__main__":
	main()