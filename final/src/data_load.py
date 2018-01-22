# -*- coding: utf-8 -*-
#/usr/bin/python2

from functools import wraps
import threading

from tensorflow.python.platform import tf_logging as logging

from params import Params
import numpy as np
import tensorflow as tf
from process import *
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Adapted from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as producer_func.
    Args:
      func: A function to decorate.
    """
    @wraps(func)
    def wrapper(inputs, dtypes, capacity, num_threads):
        r"""
        Args:
            inputs: A inputs queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """
        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(inputs))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)

        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        # create FIFO queue
        queue = tf.FIFOQueue(capacity, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        # return de-queue operation
        return queue.dequeue()

    return wrapper


class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1

def load_data(dir_, testdata = False):
    # Target indices
    if testdata:
        indices = zero_target()
    else:
        indices = load_target(dir_ + Params.target_dir)

    # Load question data
    print("Loading question data...")
    q_word_ids, q_word_len = load_word(dir_ + Params.q_word_dir)

    # Load passage data
    print("Loading passage data...")
    p_word_ids, p_word_len = load_word(dir_ + Params.p_word_dir)

    # Get max length to pad
    p_max_word = Params.max_p_len#np.max(p_word_len)
    q_max_word = Params.max_q_len#,np.max(q_word_len)

    # pad_data
    print("Preparing data...")
    p_word_ids = pad_data(p_word_ids,p_max_word)
    q_word_ids = pad_data(q_word_ids,q_max_word)
    
    # to numpy
    indices = np.reshape(np.asarray(indices,np.int32),(-1,2))
    p_word_len = np.reshape(np.asarray(p_word_len,np.int32),(-1,1))
    q_word_len = np.reshape(np.asarray(q_word_len,np.int32),(-1,1))

    for i in range(p_word_len.shape[0]):
        if p_word_len[i,0] > p_max_word:
            p_word_len[i,0] = p_max_word
    for i in range(q_word_len.shape[0]):
        if q_word_len[i,0] > q_max_word:
            q_word_len[i,0] = q_max_word

    # shapes of each data
    shapes=[(p_max_word,),(q_max_word,),
            (1,),(1,),(2,)]

    return ([p_word_ids, q_word_ids,
            p_word_len, q_word_len,
            indices], shapes)

def get_valid():
    valid, shapes = load_data(Params.valid_dir)
    indices = valid[-1]
    # valid = [np.reshape(input_, shapes[i]) for i,input_ in enumerate(valid)]

    valid_ind = np.arange(indices.shape[0],dtype = np.int32)
    np.random.shuffle(valid_ind)
    return valid, valid_ind

def get_batch(is_training = True):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load dataset
        if is_training == True:
            input_list, shapes = load_data(Params.train_dir)
        else:
            input_list, shapes = load_data(Params.test_dir, testdata = True)
        indices = input_list[-1]

        train_ind = np.arange(indices.shape[0],dtype = np.int32)
        if is_training == True:
            np.random.shuffle(train_ind)

        size = Params.data_size
        if Params.data_size > indices.shape[0] or Params.data_size == -1:
            size = indices.shape[0]
        ind_list = tf.convert_to_tensor(train_ind[:size])

        # Create Queues
        if is_training == True:
            ind_list = tf.train.slice_input_producer([ind_list], shuffle=True)
        else:
            ind_list = tf.train.slice_input_producer([ind_list], shuffle=False)
         
        @producer_func
        def get_data(ind):
            '''From `_inputs`, which has been fetched from slice queues,
               then enqueue them again.
            '''
            return [np.reshape(input_[ind], shapes[i]) for i,input_ in enumerate(input_list)]

        data = get_data(inputs=ind_list,
                        dtypes=[np.int32]*5,
                        capacity=Params.batch_size*32,
                        num_threads=6)

        # create batch queues
        batch = tf.train.batch(data,
                                shapes=shapes,
                                num_threads=2,
                                batch_size=Params.batch_size,
                                capacity=Params.batch_size*32,
                                dynamic_pad=True)

    return batch, size // Params.batch_size

def context_pointers(index, q_ind, batch = Params.batch_size):
    aaa=[0]*50###
    if len(index) != batch: batch = len(index)
    index = pointer_adjust(index)
    pointers = np.zeros((len(index), 2)).astype('int32')
    q_id = [-1] * len(index)
    with codecs.open(Params.test_dir + Params.p_len_lab,'r','utf-8') as f:
        p_len_lab = []
        for i in range(Params.num_questions): p_len_lab.append(f.readline())
    data = json.load(codecs.open(Params.test_file,"rb","utf-8",errors='replace'))
    data = data['data']
    num_context=0; num_question=0
    for topic in data:
        for para in topic['paragraphs']:
            num_context+=1
            for qas in para['qas']:
                num_question+=1
                id_ = qas['id']
                for j in range(len(q_ind)):
                    if q_ind[j] + 1 == num_question:
                        label = str.split(p_len_lab[num_question-1])
                        #print(label[-1])#######
                        #print(len(para['context']))#######
                        #print('----------')########
                        aaa[j]=para['context']#####
                        q_id[j] = id_
                        start = index[j][0]
                        stop  = index[j][1]
                        if len(label) <= stop:
                            stop = -1
                            start = -2
                        space = 0
                        for i in range(int(label[start])):
                            if para['context'][i]=='\n': space += 1
                        if stop == -1: space = 0
                        start = int(label[start]) + space
                        stop  = int(label[stop]) + space
                        pointers[j][0] = start
                        pointers[j][1] = stop
    return pointers, q_id,aaa######
                            
def pointer_adjust(index, batch = Params.batch_size):
    if len(index) != batch: batch = len(index)
    for i in range(batch):
        start = int(index[i][0])
        stop  = int(index[i][1])
        if start == stop: stop = start+1
        if start > stop: stop = start+3
        if abs(start-stop) > 4: stop = start+4
        index[i][0] = start
        index[i][1] = stop
    return index

def find_q(question):
    no_match = 0
    q_ind = [-1] * len(question)
    q_word_ids, _ = load_word(Params.test_dir + Params.q_word_dir)
    for j in range(len(question)):
        for i in range(len(q_word_ids)):
            qas = question[j][:len(q_word_ids[i])]
            if all(qas == q_word_ids[i]): q_ind[j] = i
        if q_ind[j] == -1: no_match += 1
    return q_ind, no_match

def find_id():
    data = json.load(codecs.open(Params.test_file,"rb","utf-8",errors='replace'))
    data = data['data']
    id_ = [0] * Params.num_questions
    num_context=0; num_question=0
    for topic in data:
        for para in topic['paragraphs']:
            num_context+=1
            for qas in para['qas']:
                id_[num_question] = qas['id'];
                num_question+=1
    return id_

def write_predict(q_ids, q_ans):
    q_ids = find_id()
    with open(Params.predict_dir, 'w') as f:
        f.write('id,answer\n')
        for i in range(len(q_ids)):
            start = int(q_ans[i][0])
            stop  = int(q_ans[i][1])
            id_ = q_ids[i]
            ans = ''
            for j in range(start, stop):
                if j != stop - 1:
                    ans += str(j) + ' '
                else:
                    ans += str(j)
            f.write(id_ + ',' + ans + '\n')
    
def pad_data(data, max_word):
    padded_data = np.zeros((len(data),max_word),dtype = np.int32)
    for i,line in enumerate(data):
        for j,word in enumerate(line):
            if j >= max_word:
                break
            padded_data[i,j] = word
    return padded_data

def load_target(dir):
    data = []
    count = 0
    with codecs.open(dir,"rb","utf-8") as f:
        line = f.readline()
        while count < 1000 if Params.mode == "debug" else line:
            line = [int(w) for w in line.split()]
            data.append(line)
            count += 1
            line = f.readline()
    return data

def zero_target():
    data = []
    for i in range(Params.num_questions):
        line = [0,0]
        data.append(line)
    return data

def load_word(dir):
    data = []
    w_len = []
    count = 0
    with codecs.open(dir,"rb","utf-8") as f:
        line = f.readline()
        while count < 1000 if Params.mode == "debug" else line:
            line = [int(w) for w in line.split()]
            data.append(line)
            count += 1
            w_len.append(len(line))
            line = f.readline()
    return data, w_len

def max_value(inputlist):
    max_val = 0
    for list_ in inputlist:
        for val in list_:
            if val > max_val:
                max_val = val
    return max_val
