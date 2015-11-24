import dill as pickle
import numpy as np
from incremental_k_means import Cluster

def cos_similarity(cl1, cl2):
	return np.dot(cl1, cl2)/np.linalg.norm(cl1)/np.linalg.norm(cl2)

def no_of_docs_similarity(cl1, cl2):
	#print cl1.doc_grp.keys(), '\n', cl2.doc_grp.keys()
	common = set(cl1.doc_grp.keys()).intersection(set(cl2.doc_grp.keys()))
	return len(common)

def compute_best(actual_clusters, gen_clusters, l_ac, l_gen):
	matched_clusters = np.empty(l_gen, dtype = int)
	for i in xrange(l_gen):
		#max_sim = cos_similarity(actual_clusters[0].centroid, gen_clusters[i].centroid)
		max_common = no_of_docs_similarity(actual_clusters[0], gen_clusters[i])
		max_index = 0
		for j in xrange(l_ac):
			#sim = cos_similarity(actual_clusters[j].centroid, gen_clusters[i].centroid)
			no_of_common = no_of_docs_similarity(actual_clusters[j], gen_clusters[i])
			if (no_of_common > max_common):
				max_common = no_of_common
				max_index = j
		matched_clusters[i] = max_index
		# print max_sim, "\t", set(actual_clusters[matched_clusters[i]].documents.keys()).intersection(set(gen_clusters[i].documents.keys()))
	return matched_clusters

def compute_f_measure(actual_clusters, gen_clusters, l_ac, l_gen):
	matched_clusters = compute_best(actual_clusters, gen_clusters, l_ac, l_gen)
	precision = np.empty(l_gen, dtype = float)
	recall = np.empty(l_gen, dtype = float)
	f_measure = np.empty(l_gen, dtype = float)
	for i in xrange(l_gen):
		docs_gen = gen_clusters[i].doc_grp.keys()
		docs_act = actual_clusters[matched_clusters[i]].doc_grp.keys()
		common = len(set(docs_gen).intersection(set(docs_act)))
		# print "actual\t", sorted(docs_act)
		# print "found\t", sorted(docs_gen)
		if common == 0:
			f_measure[i] = 0.
			precision[i] = 0.
			recall[i]    = 0.
		else:
			precision[i] = float(common)/gen_clusters[i].length
			recall[i] = float(common)/actual_clusters[matched_clusters[i]].length
			f_measure[i] = 2*precision[i]*recall[i]/(precision[i] + recall[i])
	return precision,recall, f_measure

def main():
	clusters = pickle.load(open("clusters_first_post.pickle", "rb"))
	actual_clusters = clusters['reference']; l_ac = len(actual_clusters)
	gen_clusters    = clusters['generated']; l_gen = len(gen_clusters)

	# for cluster in gen_clusters:
	# 	cluster.centroid = np.array(cluster.documents.values()).mean(axis=0)

	pre, rec, f_measure = compute_f_measure(actual_clusters, gen_clusters, l_ac, l_gen)
	non_nan = list()
	for f in f_measure:
		if f:
			non_nan.append(f)

	R = rec.mean()
	P = pre.mean()
	print "Recall        \t", R
	print "Precision     \t", P
	print "Cluster-F Mean\t", sum(non_nan) / len(f_measure)
	print "Average-F Mean\t", (2*R*P) / (R+P)

	rec_ = list()
	pre_ = list()
	for i in xrange(l_gen):
		if rec[i] != 0. and pre[i] != 0. :
			rec_.append(rec[i])
			pre_.append(pre[i])
	print 
	print "Non Nan"
	R = np.array(rec_).mean()
	P = np.array(pre_).mean()
	print "Recall        \t", R
	print "Precision     \t", P
	print "Average-F Mean\t", (2*R*P) / (R+P)

if __name__ == "__main__":
	main()