import numpy as np 
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import collections


attn_dir = '../../results/hops_5_l2_0.0_embed_size_20_glove_false_nonlinearity_None/'
pred_dir = attn_dir
truth_dir = attn_dir
# attn_dir = '../../results/tmp/single_sharing_no_encode/'
# pred_dir = '../../results/tmp/single_sharing_no_encode/'
# truth_dir = '../../results/tmp/single_sharing_no_encode/'
# attn_dir = '../../results/tmp/single_sharing/'
# pred_dir = '../../results/tmp/single_sharing/'
# truth_dir = '../../results/tmp/single_sharing/'
# attn_dir = '../../results/tmp/single_sharing_relu_wencode/'
# pred_dir = '../../results/tmp/single_sharing_relu_wencode/'
# truth_dir = '../../results/tmp/single_sharing_relu_wencode/'
data_dir = '../../data/tasks_1-20_v1-2/en'
task_id = 8

def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
            # num_questions = 0
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            supporting = map(int, supporting.split())
            supporting = [i-1 for i in supporting]
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                #substory = [x for x in story if x]
                #print substory
                substory = [x for x in story]

            data.append((substory, q, a, supporting))
            # num_questions += 1
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def heatmap_attention(mat_in, data):
	"""mat_in must be num_sentences x num_layers
		data is of the form (story, question, answer, supporting_fact)"""
	f = []
	ylabel = data[0]
	for sent in ylabel:
		if sent:
			s = ' '.join(sent)
			f.append(s)
	question = ' '.join(data[1]) + '?'
	answer = ' '.join(data[2])
	plt.title(question + ' ' + answer)
	ax = sns.heatmap(mat_in, vmin=0, vmax=1, yticklabels=f, annot=True)
	plt.tight_layout()
	plt.show()

def error_by_type(misclassified, attn_mat, test_data):
	imagination, okay_memory, poor_memory, misinterpret = 0,0,0,0
	total_wrong = float(len(misclassified))
	for i in misclassified:
		attn_win_per_hop = np.argmax(attn_mat[:,i,:], axis=1)
		attn_win = attn_win_per_hop[-1]
		# print 'Length: '+ str(len(test_data[i][0]))
		# print  'attn: ' + str(attn_win)
		if attn_win >= len(test_data[i][0]):
			imagination += 1
		elif attn_win not in test_data[i][3]:
			poor_memory += 1
		elif truth_mat[i] not in test_data[i][0][attn_win]:
			okay_memory += 1
		elif attn_win in test_data[i][3]:
			misinterpret += 1
	print('Imagine: '+str((imagination/total_wrong)*100))
	print('PoorMemory: '+str((poor_memory/total_wrong)*100))
	print('OkayMemory: '+str((okay_memory/total_wrong)*100))
	print('Misinterpret: '+str((misinterpret/total_wrong)*100))
	return [imagination, okay_memory, poor_memory, misinterpret]

def vis_attn(attn_mat, test_data, wrong_idx, print_lim=10):
	print_min = 5
	visualize_wrong = [c for (c,(i,j,k,l)) in enumerate(test_data) if len(i)<=print_lim and len(i)>=print_min and c in wrong_idx]

	vis_id = visualize_wrong[np.random.choice(range(len(visualize_wrong)),1)]
	print visualize_wrong
	print(test_data[vis_id])
	print(predicted_mat[vis_id])
	print vis_id
	#vis_id = 355
	f = []
	ylabel = test_data[vis_id][0]
	for sent in ylabel:
		if sent:
			s = ' '.join(sent)
			f.append(s)
	plt_id_max = min(print_lim, len(f))
	mat_in = attn_mat[:, vis_id, :plt_id_max].transpose()
	heatmap_attention(mat_in, test_data[vis_id])

def error_by_sl(misclassified, test_data):
	sl_wrong_freq = collections.defaultdict(int)
	sl_all = collections.defaultdict(int)
	for i in misclassified:
		f = []
		for s in test_data[i][0]:
			if s:
				f.append(s)
		sl_wrong_freq[len(f)] += 1
	for i in range(0, len(test_data)):
		f = []
		for s in test_data[i][0]:
			if s:
				f.append(s)
		sl_all[len(f)] += 1
	print sl_all
	print sl_wrong_freq
	x_axis = []
	y_axis = []
	for key in sorted(sl_wrong_freq.iterkeys()):
		x_axis.append(key)
		y_axis.append(sl_wrong_freq[key])
	plt.plot(x_axis, y_axis)
	x_axis = []
	y_axis = []
	for key in sorted(sl_all.iterkeys()):
		x_axis.append(key)
		y_axis.append(sl_all[key])
	plt.plot(x_axis, y_axis)
	plt.show()

if __name__ == '__main__':
	attn_mat = np.load(os.path.join(attn_dir,'task_'+str(task_id)+'_attention.npy'))
	predicted_mat = np.load(os.path.join(pred_dir,'task_'+str(task_id)+'_pred.npy'))
	truth_mat = np.load(os.path.join(truth_dir,'task_'+str(task_id)+'_truth.npy'))
	idx = predicted_mat!=truth_mat
	idx = [c for c,v in enumerate(idx) if v == True]
	corr_idx = predicted_mat==truth_mat
	corr_idx = [c for c,v in enumerate(corr_idx) if v == True]
	train_data, test_data = load_task(data_dir, task_id)
	misclassified = [c for (c,(i,j,k,l)) in enumerate(test_data) if c in idx]
	print len(misclassified)
	if len(misclassified) > 0:
		# Cause of error
		errors = error_by_type(misclassified, attn_mat, test_data)
		# Error frequency by sentence length
		error_by_sl(misclassified, test_data)
		# Visualize errors
		vis_attn(attn_mat, test_data, idx)

	# Visualize truth
	vis_attn(attn_mat, test_data, corr_idx)

	# Did layer 0 attention pick the right candidate?
	corr_attn = 0.0
	for i in range(len(test_data)):
		attn_win_per_hop = np.argmax(attn_mat[:,i,:], axis=1)
		attn_win = attn_win_per_hop[0]
		f = []
		for j in test_data[i][0]:
			if j:
				f.append(j)
		if attn_win < len(f):
			sen_win = f[attn_win]
			query = test_data[i][1]
			# print sen_win
			# print query
			for w in sen_win:
				if w in query and w not in ['the', 'is']:
					corr_attn += 1
					break
	print('Correct attn_layer0: ' + str((corr_attn/len(test_data))*100))

	corr_attn = 0.0
	examples = []
	for i in range(len(test_data)):
		attn_win_per_hop = np.argmax(attn_mat[:,i,:], axis=1)
		attn_win_0 = attn_win_per_hop[0]
		attn_win_last = attn_win_per_hop[-1]
		f = []
		for j in test_data[i][0]:
			if j:
				f.append(j)
		if attn_win_0 < len(f) and attn_win_last < len(f):
			sen_win_0 = f[attn_win_0]
			sen_win_last = f[attn_win_last]
			query = test_data[i][1]
			answer = test_data[i][2]
			# print sen_win_0
			# print sen_win_last
			# print query
			for w in sen_win_0:
				if w in query and w not in ['the', 'is']:
					for ans in sen_win_last:
						if ans in answer:
							corr_attn += 1
							examples.append(i)
							break
	print examples
	print('Correct attn_layer0_last: ' + str((corr_attn/len(test_data))*100))

	football = 0
	fp = 0
	tp = 0
	fn = 0
	tn = 0
	posclass = 'hallway'
	for i in range(len(test_data)):
		if 'football' in test_data[i][1]:
			football += 1
			print test_data[i][1]
			print predicted_mat[i]
			print truth_mat[i]
			if posclass in truth_mat[i] and posclass in predicted_mat[i]:
				tp += 1
			elif posclass not in truth_mat[i] and posclass not in predicted_mat[i]:
				tn += 1
			elif posclass in truth_mat[i] and posclass not in predicted_mat[i]:
				fn += 1
			elif posclass not in truth_mat[i] and posclass in predicted_mat[i]:
				fp += 1
	print tp/float(tp+fp)
	print tp/float(tp+fn)
	print football