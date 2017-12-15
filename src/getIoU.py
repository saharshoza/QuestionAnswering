'''
calculate IoU from finished runs
'''
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import os, sys, re
import pandas as pd
from sets import Set


def calculate_iou(attention_hops, examples, hops):
    iou = [0.0] * hops
    stopwords = Set(
        ['the', 'is', 'was', 'what', 'where', 'in', 'up', 'to', 'a', 'an', 'and', 'are', 'as', 'at', 'be', 'for',
         'from'])

    def iou_hop(story, query, answer, example_number):
        # print('query: ' + str(query))
        # print('story: ' + str(story))
        qset = Set(query) - stopwords
        aset = Set(answer) - stopwords
        # print('query words set: ' + str(qset))
        sset_list = [Set(sentence) - stopwords for sentence in story]
        # print('sentences words list ' + str(sset_list))
        for hop_iter in range(hops):
            # if hop_iter == FLAGS.hops-1:
            #     qset = aset
            matchedidxset = Set([])
            matchedwordsset = Set([])
            for idx, val in enumerate(sset_list):
                if bool(qset.intersection(val)):
                    # print('NULLLLLLLL')
                    matchedidxset.add(idx)
                    matchedwordsset = matchedwordsset.union(val)

            # print('matched index set: ' + str(matchedidxset))
            # print('matched words set: ' + str(matchedwordsset))
            # print(attention_hops[0].shape)
            attention_order = np.argsort(attention_hops[hop_iter][example_number])
            # print(attention_order[:-len(matchedidxset)].shape)
            attention_winners_set = Set(list(attention_order[-len(matchedidxset):]))
            # print('attention winners set: ' + str(attention_winners_set))
            # print('numerator: ' + str(len(matchedidxset.intersection(attention_winners_set))))
            # print('denom: ' + str(float(len(matchedidxset.union(attention_winners_set)))))

            iou[hop_iter] += (
                len(matchedidxset.intersection(attention_winners_set)) / float(len(matchedidxset.union(attention_winners_set))))
            qset = matchedwordsset - qset
            # print('query words set at hop ' + str(hop_iter) + ': ' + str(qset))

    for index, ex in enumerate(examples):
        s, q, a = ex
        iou_hop(s, q, a, index)
        # print('IoU at example ' + str(iou))

    return iou

if __name__ == '__main__':

    Tasks = [6,7,10,17,18,19]
    data_dir = "data/tasks_1-20_v1-2/en/"
    logs_dir = "logs/"

    directories = os.walk(logs_dir).next()[1]
    directories = [logs_dir + directory + '/' for directory in directories if 'LeakyRelu' not in directory]
    new_IoUs = []
    for task_id in Tasks:
        train, test = load_task(data_dir, task_id)
        task_IoUs = []
        for directory in directories:
            m = re.search('(hops_(\d)_)', directory)
            NoOfhops = int(m.group(2))
            task_attention_file = directory + 'task_' + str(task_id) + '_attention.npy'
            attention_hops = np.load(task_attention_file)
            dir_iou = calculate_iou(attention_hops, test, NoOfhops)[-1]
            task_IoUs.append(dir_iou)

        new_IoUs.append(task_IoUs)

    for dir_idx, directory in enumerate(directories):
        output_csv_file = directory + 'stats_new.csv'
        data = pd.read_csv(output_csv_file)
        last_column = data.columns.tolist()[-1]

        stats = list(data[last_column])

        for task_idx, task_id in enumerate(Tasks):
            stats[task_id-1] = new_IoUs[task_idx][dir_idx]
        stats = [float("{0:.2f}".format(i)) for i in stats]

        df = pd.DataFrame({
            last_column: stats
        }, columns=[last_column])

        df.to_csv(output_csv_file, index=False)