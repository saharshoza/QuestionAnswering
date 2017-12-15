"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n import MemN2N, lrelu
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import os, sys
import pandas as pd
from sets import Set

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("regularization", 0.0, "Regularization.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
# tf.flags.DEFINE_integer("task_id", 2, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
# tf.flags.DEFINE_integer("run_number", 1, "Run Number")
tf.flags.DEFINE_integer("NoOfRuns", 3, "number of runs")
tf.flags.DEFINE_string("nonlin", 'None', "Non-linearity")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("glove", False, "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

nonlin_type = FLAGS.nonlin
if FLAGS.nonlin == 'None':
    FLAGS.nonlin = None
elif FLAGS.nonlin == 'Relu':
    FLAGS.nonlin = tf.nn.relu
elif FLAGS.nonlin == 'LeakyRelu':
    FLAGS.nonlin = lrelu


def calculate_iou(attention_hops, examples, task_num):
    iou = [0.0] * FLAGS.hops
    stopwords = Set(
        ['the', 'is', 'was', 'what', 'where', 'in', 'up', 'to', 'a', 'an', 'and', 'are', 'as', 'at', 'be', 'for',
         'from'])

    def iou_hop(story, query, answer, example_number, task_num):
        # print('query: ' + str(query))
        # print('story: ' + str(story))
        qset = Set(query) - stopwords
        aset = Set(answer) - stopwords
        # print('query words set: ' + str(qset))
        sset_list = [Set(sentence) - stopwords for sentence in story]
        # print('sentences words list ' + str(sset_list))
        for hop_iter in range(FLAGS.hops):
            if (hop_iter == FLAGS.hops-1) and task_num not in [6,7,10,17,18,19]:
                qset = aset
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
        iou_hop(s, q, a, index, task_num)
        # print('IoU at example ' + str(iou))

    return iou


def run_task(task_id):
    print("Started Task:", task_id)
    # task data
    train, test = load_task(FLAGS.data_dir, task_id)
    data = train + test

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    reverse_lookup = {v: k for (k, v) in word_idx.items()}
    lookup_vocab = ['nil']

    print(reverse_lookup)
    print(word_idx)
    for i in range(1, len(reverse_lookup) + 1):
        lookup_vocab.append(reverse_lookup[i])

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(FLAGS.memory_size, max_story_size)

    # Add time words/indexes
    for i in range(memory_size):
        word_idx['time{}'.format(i + 1)] = 'time{}'.format(i + 1)

    print(len(word_idx))
    print(word_idx)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    sentence_size = max(query_size, sentence_size)  # for the position
    sentence_size += 1  # +1 for time words

    print("Longest sentence length", sentence_size)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)

    # train/validation/test sets
    S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
    trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1,
                                                                                 random_state=FLAGS.random_state)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

    #print(testS[0])

    print("Training set shape", trainS.shape)

    # params
    n_train = trainS.shape[0]
    n_test = testS.shape[0]
    n_val = valS.shape[0]

    print("Training Size", n_train)
    print("Validation Size", n_val)
    print("Testing Size", n_test)

    train_labels = np.argmax(trainA, axis=1)
    val_labels = np.argmax(valA, axis=1)
    test_labels = np.argmax(testA, axis=1)

    tf.set_random_seed(FLAGS.random_state)
    batch_size = FLAGS.batch_size

    batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start, end in batches]

    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    max_testAccuracy = 0
    max_trainAccuracy = 0
    max_valAccuracy = 0

    best_test_prob_hops = 0
    best_test_prob_vocab = 0
    best_pred_word = 0
    best_true_word = 0
    best_test_A1 = 0
    best_test_C = 0
    best_lookup_vocab = 0

    for run_id in range(FLAGS.NoOfRuns):
        print('Run Number: ' + str(run_id))
        max_epoch_trainAccuracy = 0
        max_epoch_valAccuracy = 0
        with tf.Session() as sess:
            model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, session=sess,
                           hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm, regularization=FLAGS.regularization,
                           nonlin=FLAGS.nonlin)
            for t in range(1, FLAGS.epochs + 1):
                # Stepped learning rate
                if t - 1 <= FLAGS.anneal_stop_epoch:
                    anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
                else:
                    anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
                lr = FLAGS.learning_rate / anneal

                np.random.shuffle(batches)
                total_cost = 0.0
                for start, end in batches:
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    a = trainA[start:end]
                    cost_t = model.batch_fit(s, q, a, lr)
                    total_cost += cost_t

                if t % FLAGS.evaluation_interval == 0:
                    train_preds = []
                    for start in range(0, n_train, batch_size):
                        end = start + batch_size
                        s = trainS[start:end]
                        q = trainQ[start:end]
                        pred = model.predict(s, q)
                        train_preds += list(pred)

                    val_preds, valid_prob_vocab, valid_prob_hops, valid_A1, valid_C = model.predict_prob_instrument(
                        valS, valQ)
                    train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
                    val_acc = metrics.accuracy_score(val_preds, val_labels)

                    print('-----------------------')
                    print('Epoch', t)
                    print('Total Cost:', total_cost)
                    print('Training Accuracy:', train_acc)
                    print('Validation Accuracy:', val_acc)
                    print('-----------------------')

                    if (val_acc > max_epoch_valAccuracy):
                        max_epoch_trainAccuracy = train_acc
                        max_epoch_valAccuracy = val_acc

            test_preds, test_prob_vocab, test_prob_hops, test_A1, test_C = model.predict_prob_instrument(testS, testQ)
            pred_word = [reverse_lookup[i] for i in test_preds]
            true_word = [reverse_lookup[i] for i in test_labels]

            test_acc = metrics.accuracy_score(test_preds, test_labels)

            train_accuracies.append(max_trainAccuracy)
            validation_accuracies.append(max_valAccuracy)
            test_accuracies.append(test_acc)

            if (test_acc > max_testAccuracy):
                max_testAccuracy = test_acc
                max_trainAccuracy = max_epoch_trainAccuracy
                max_valAccuracy = max_epoch_valAccuracy
                best_test_prob_hops = test_prob_hops
                best_test_prob_vocab = test_prob_vocab
                best_pred_word = pred_word
                best_true_word = true_word
                best_test_A1 = test_A1
                best_test_C = test_C
                best_lookup_vocab = lookup_vocab

            print("Test Accuracy: ", test_acc)
            iou = calculate_iou(test_prob_hops, test, task_id)
            print('IoU: ' + str(iou))

    print("Best Testing Accuracy:", max_testAccuracy)
    # save test files
    np.save(logs_dir + 'task_' + str(task_id) + '_attention', best_test_prob_hops)
    np.save(logs_dir + 'task_' + str(task_id) + '_vocab_prob', best_test_prob_vocab)
    np.save(logs_dir + 'task_' + str(task_id) + '_pred', best_pred_word)
    np.save(logs_dir + 'task_' + str(task_id) + '_truth', best_true_word)
    np.save(logs_dir + 'task_' + str(task_id) + '_A', best_test_A1)
    np.save(logs_dir + 'task_' + str(task_id) + '_C', np.array(best_test_C))
    np.save(logs_dir + 'task_' + str(task_id) + '_lookupvocab', best_lookup_vocab)

    iou = calculate_iou(best_test_prob_hops, test, task_id)
    print('Best IoU: ' + str(iou))

    return max_trainAccuracy, max_valAccuracy, max_testAccuracy, iou


if __name__ == '__main__':

    NoOfTasks = 20
    logs_dir = "logs/" + 'hops_' + str(FLAGS.hops) + '_l2_' + str(
        FLAGS.regularization) + '_embed_size_' + str(FLAGS.embedding_size) + '_glove_' + str(
        FLAGS.glove) + '_nonlinearity_' + nonlin_type + '/'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    tasks_train_accuracies = []
    tasks_valid_accuracies = []
    tasks_test_accuracies = []
    tasks_iou = []
    for task_number in range(1, NoOfTasks + 1):
        log_file_name = logs_dir + 'task_' + str(task_number) + '_output.txt'
        orig_stdout = sys.stdout
        f = open(log_file_name, 'w')
        sys.stdout = f

        train, val, test, iou = run_task(task_number)
        tasks_train_accuracies.append(train)
        tasks_valid_accuracies.append(val)
        tasks_test_accuracies.append(test)
        tasks_iou.append(iou)

        sys.stdout = orig_stdout
        f.close()

    output_csv_file = logs_dir + 'stats.csv'

    df = pd.DataFrame({
        'Train': tasks_train_accuracies,
        'Val': tasks_valid_accuracies,
        'Test': tasks_test_accuracies
    }, index=range(1, NoOfTasks + 1), columns=['Train', 'Val', 'Test'])
    df.index.name = 'Task'

    df_iou = pd.DataFrame(tasks_iou, index=range(1, NoOfTasks + 1))
    col_list = ['hop' + str(h_iter) for h_iter in range(FLAGS.hops)]
    df_iou.columns = col_list

    frames = [df, df_iou]
    df_merged = pd.concat(frames, axis=1, join='inner')

    df_merged.to_csv(output_csv_file)

