import numpy as np 
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict
from sets import Set
from visualization import load_task
import pandas as pd
from sklearn.manifold import TSNE


log_dir = '../../results/hops_3_l2_0.0_embed_size_20_glove_false_nonlinearity_None/'
data_dir = '../../data/tasks_1-20_v1-2/en'
task_id = 19
plot = False


def f1_score(confusion_mat):
	precision_denom = np.sum(confusion_mat, axis=0)
	recall_denom = np.sum(confusion_mat, axis=1)
	numerator = np.array([confusion_mat[i][i] for i in range(confusion_mat.shape[0])])
	precision = numerator/precision_denom
	recall = numerator/recall_denom
	#print precision, 	recall
	f1 = (precision*recall)/(precision+recall)
	print f1
	return f1

def print_confusion_matrix(confusion_mat, answers, query):
	confusion_mat = confusion_mat.astype('int')
	df_cm = pd.DataFrame(confusion_mat, index = [i for i in answers],
	                  columns = [i for i in answers])
	plt.figure(figsize = (10,7))
	plt.title(query)
	sns.heatmap(df_cm, annot=True, fmt="d")
	plt.show()

def confusion_question(truth_mat, predicted_mat, answer_map):
	confusion_mat = np.zeros((len(answer_map), len(answer_map)))
	for i in range(len(truth_mat)):
		true_idx = answer_map[truth_mat[i]]
		pred_idx = answer_map[predicted_mat[i]]
		confusion_mat[true_idx][pred_idx] += 1
	print confusion_mat	
	f1_score(confusion_mat)
	return confusion_mat

def print_embedding_mat(embed_mat, intermediate_mat, yvocab):
	ax = sns.heatmap(embed_mat[:len(yvocab)], yticklabels=yvocab)
	plt.show()
	#a = np.sum(np.abs(embed_mat[:len(yvocab)]), axis=1)
	# a = np.sum(pow(embed_mat[:len(yvocab)], 2), axis=1)
	a = np.linalg.norm(embed_mat[:len(yvocab)], axis=1)
	sorted_idx = np.argsort(a)
	sorted_vocab = yvocab[sorted_idx]

	norm_out = np.vstack((yvocab, a))

	for hop in range(intermediate_mat.shape[0]):
		print hop
		# a = np.sum(pow(intermediate_mat[hop][:len(yvocab)], 2), axis=1)
		a = np.linalg.norm(intermediate_mat[hop][:len(yvocab)], axis=1)
		sorted_idx = np.argsort(a)
		sorted_vocab = yvocab[sorted_idx]
		print sorted_vocab
		norm_out = np.vstack((norm_out, a))
		ax = sns.heatmap(intermediate_mat[hop][:len(yvocab)], yticklabels=yvocab)
		plt.show()

	# df = pd.DataFrame({
	# 'vocab': sorted_vocab,
	# 'val': np.sort(a)
	# })
	df = pd.DataFrame(norm_out.transpose())
	df.to_csv('norm.out')

	# print sorted_vocab
	# print np.sort(a)
	# print a
	# tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=500)
	# tsne_results = tsne.fit_transform(embed_mat[:len(yvocab)])
	# print tsne_results.shape
	# plt.scatter(tsne_results[:,0], tsne_results[:,1])
	# for idx, word in enumerate(yvocab):
	# 	plt.annotate(word, (tsne_results[idx, 0], tsne_results[idx, 1]))
	# plt.show()

if __name__ == '__main__':
	attn_mat = np.load(os.path.join(log_dir,'task_'+str(task_id)+'_attention.npy'))
	predicted_mat = np.load(os.path.join(log_dir,'task_'+str(task_id)+'_pred.npy'))
	truth_mat = np.load(os.path.join(log_dir,'task_'+str(task_id)+'_truth.npy'))
	embed_mat = np.load(os.path.join(log_dir, 'task_' + str(task_id) + '_A.npy'))
	intermediate_mat = np.load(os.path.join(log_dir, 'task_' + str(task_id) + '_C.npy'))
	yvocab = np.load(os.path.join(log_dir,'task_' + str(task_id) + '_lookupvocab.npy'))

	# Load dataset
	_, test_data = load_task(data_dir, task_id)

	# Find unique queries
	# Find idx corresponding to query
	query_map = defaultdict(list)
	answer_set = Set()
	for idx, ex in enumerate(test_data):
		s, q, a, sf = ex
		q_str = ' '.join(q)
		answer_set.add(a[0])
		query_map[q_str].append(idx)
	for val in predicted_mat:
		answer_set.add(val)
	answer_map = dict((val, idx) for idx, val in enumerate(answer_set))
	# Compute confusion matrix
	net_confusion_mat = np.zeros((len(answer_map), len(answer_map)))
	print answer_map
	for k in query_map.keys():
		print k
		true = truth_mat[query_map[k]]
		pred = predicted_mat[query_map[k]]
		confusion_mat_question = confusion_question(true, pred, answer_map)
		if plot:
			print_confusion_matrix(confusion_mat_question, answer_map.keys(), k+'?')
		net_confusion_mat += confusion_mat_question
	f1 = f1_score(net_confusion_mat)
	if plot:
		print_confusion_matrix(net_confusion_mat, answer_map.keys(), 'AllQuestions')
	print np.mean(f1)

	print_embedding_mat(embed_mat, intermediate_mat, yvocab)
