from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce
from collections import defaultdict
import tensorflow as tf
import numpy as np
import os, sys, re
import pandas as pd
from sets import Set
from data_utils import load_task, vectorize_data
import math

data_dir = "data/tasks_1-20_v1-2/en"
plot = False
def f1_score(confusion_mat):
    precision_denom = np.sum(confusion_mat, axis=0)
    recall_denom = np.sum(confusion_mat, axis=1)
    numerator = np.array([confusion_mat[i][i] for i in range(confusion_mat.shape[0])])
    precision = numerator / precision_denom
    recall = numerator / recall_denom
    # precision = np.array([i  if not math.isnan(i) else 0 for i in precision])
    # recall = np.array([i if not math.isnan(i) else 0  for i in recall])
    # print(precision)
    # print(recall)
    f1 = (precision * recall) / (precision + recall)
    return f1


def confusion_question(truth_mat, predicted_mat, answer_map):
    confusion_mat = np.zeros((len(answer_map), len(answer_map)))
    for i in range(len(truth_mat)):
        true_idx = answer_map[truth_mat[i]]
        pred_idx = answer_map[predicted_mat[i]]
        confusion_mat[true_idx][pred_idx] += 1
    f1_score(confusion_mat)
    return confusion_mat


def get_F1(log_dir, task_id):
    predicted_mat = np.load(os.path.join(log_dir, 'task_' + str(task_id) + '_pred.npy'))
    truth_mat = np.load(os.path.join(log_dir, 'task_' + str(task_id) + '_truth.npy'))

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
    for k in query_map.keys():
        true = truth_mat[query_map[k]]
        pred = predicted_mat[query_map[k]]
        confusion_mat_question = confusion_question(true, pred, answer_map)
        if plot:
            print_confusion_matrix(confusion_mat_question, answer_map.keys(), k + '?')
        net_confusion_mat += confusion_mat_question
    #print(net_confusion_mat)
    f1 = f1_score(net_confusion_mat)
    f1 = [i for i in f1 if not math.isnan(i)]
    return np.mean(f1)

if __name__ == '__main__':

    Tasks = range(0,20)
    logs_dir = "logs/"

    output_file = 'best_stats.csv'

    directories = os.walk(logs_dir).next()[1]
    directories = [logs_dir + directory + '/' for directory in directories]

    best_train = [0]*len(Tasks)
    best_val = [0]*len(Tasks)
    best_test = [0]*len(Tasks)
    best_directory = [0]*len(Tasks)
    best_f1 = [0.0]*len(Tasks)

    for directory in directories:
        stats_csv_file = directory + 'stats.csv'
        data = pd.read_csv(stats_csv_file)
        train = list(data['Train'])
        val = list(data['Val'])
        test = list(data['Test'])
        for task in Tasks:
            if(best_test[task] < test[task]):
                best_test[task] = test[task]
                best_train[task] = train[task]
                best_val[task] = val[task]
                best_directory[task] = directory
                best_f1[task] = get_F1(directory, task+1)
                if task==2:
                    print('best_f1: ' + str(best_f1[task]))


    best_train = [float("{0:.2f}".format(i)) for i in best_train]
    best_val = [float("{0:.2f}".format(i)) for i in best_val]
    best_test = [float("{0:.2f}".format(i)) for i in best_test]
    best_f1 = [float("{0:.2f}".format(float(i))) for i in best_f1]

    df = pd.DataFrame({
        'Train': best_train,
        'Val' : best_val,
        'Test' : best_test,
        'Directory' : best_directory,
        'F1' : best_f1
    }, columns=['Train', 'Val', 'Test', 'F1'])

    df.to_csv(output_file, index=False)