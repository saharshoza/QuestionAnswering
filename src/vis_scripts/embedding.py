import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from visualization import load_task

#embedding_dir  = "../../results/tmp/single_sharing"
embedding_dir  = "../../results/tmp/small_glove_norm"
#embedding_dir  = "../../results/tmp/small_glove_embeddings"
data_dir = '../../data/tasks_1-20_v1-2/en'
#attn_dir = '../../results/tmp/small_glove_embeddings/'
#attn_dir = '../../results/tmp/single_sharing/'
attn_dir = '../../results/tmp/small_glove_norm/'
pred_dir = attn_dir
truth_dir = attn_dir
task_id = 1
#task_id = 2
sentence_size = 50	
#sentence_size = 6
#embedding_size = 20
embedding_size = 50


attn_mat = np.load(os.path.join(attn_dir,'attention_task'+str(task_id)+'.npy'))
predicted_mat = np.load(os.path.join(pred_dir,'pred_task'+str(task_id)+'.npy'))
truth_mat = np.load(os.path.join(truth_dir,'truth_task'+str(task_id)+'.npy'))

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

embed_mat = np.load(os.path.join(embedding_dir, 'A_task' + str(task_id) + '.npy'))
yvocab = np.load(os.path.join(embedding_dir,'lookupvocab_task' + str(task_id) + '.npy'))
_, test_data = load_task(data_dir, task_id)
vis_id = 1


example  = test_data[vis_id]
print example
print predicted_mat[vis_id]
story, query, answer, supporting_fact = example
query = query + ["nil"]*(sentence_size-len(query))
vocab_dict = {word:idx for idx, word in enumerate(yvocab)}

encoding_mat = position_encoding(sentence_size, embedding_size)

def sentence_embed_fn(sentence, vocab_dict, pad_size, embed_mat, encoding_mat):
	sentence = sentence + ["nil"]*(pad_size-len(sentence))
	sentence_embed = []
	for s in sentence:
		sentence_embed.append(embed_mat[vocab_dict[s]])
	sentence_embed = np.array(sentence_embed)
	sentence_encode = sentence_embed * encoding_mat
	return np.sum(sentence_encode, axis=0)

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

def vis_attn(attn_mat, test_data, wrong_idx, print_lim=10):
	print_min = 5
	visualize_wrong = [c for (c,(i,j,k,l)) in enumerate(test_data) if len(i)<=print_lim and len(i)>=print_min and c in wrong_idx]

	vis_id = visualize_wrong[np.random.choice(range(len(visualize_wrong)),1)]
	# print visualize_wrong
	# print(test_data[vis_id])
	# print(predicted_mat[vis_id])
	vis_id = 1
	f = []
	ylabel = test_data[vis_id][0]
	for sent in ylabel:
		if sent:
			s = ' '.join(sent)
			f.append(s)
	plt_id_max = min(print_lim, len(f))
	mat_in = attn_mat[:, vis_id, :plt_id_max].transpose()
	heatmap_attention(mat_in, test_data[vis_id])

# Query
q_final = sentence_embed_fn(query, vocab_dict, sentence_size, embed_mat, encoding_mat)

# Sentence
for s in story:
	if s:
		print s
		s_final = sentence_embed_fn(s, vocab_dict, sentence_size, embed_mat, encoding_mat)
		print np.dot(q_final, s_final)

# Visualize attention
vis_attn(attn_mat, test_data, range(len(test_data)))
ax = sns.heatmap(embed_mat, yticklabels=yvocab)
plt.show()