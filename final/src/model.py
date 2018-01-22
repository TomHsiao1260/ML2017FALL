# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, get_valid, context_pointers, find_q, write_predict
from params import Params
from layers import *
from GRU import gated_attention_Wrapper, GRUCell, SRUCell
from evaluate import *
import numpy as np
import pickle
from process import *
import codecs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
			"adam":tf.train.AdamOptimizer,
			"gradientdescent":tf.train.GradientDescentOptimizer,
			"adagrad":tf.train.AdagradOptimizer}

class Model(object):
    def __init__(self,is_training = True):
        # Build the computational graph when initializing
        self.is_training = is_training
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.data, self.num_batch = get_batch(is_training = is_training)
            #print(self.data);print(1)
            (self.passage_w,
             self.question_w,
             self.passage_w_len_,
             self.question_w_len_,
             self.indices) = self.data

            self.passage_w_len = tf.squeeze(self.passage_w_len_)
            self.question_w_len = tf.squeeze(self.question_w_len_)

            self.encode_ids()
            self.params = get_attn_params(Params.attn_size, initializer = tf.contrib.layers.xavier_initializer)
            #print(self.params);print(12);
            self.attention_match_rnn()
            self.bidirectional_readout()
            self.pointer_network()

            if is_training:
                self.loss_function()
                self.summary()
                self.init_op = tf.global_variables_initializer()
            else:
                self.outputs()
            total_params()

    def encode_ids(self):
        with tf.device('/cpu:0'):
            self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.vocab_size, Params.emb_size]),trainable=False, name="word_embeddings")
            self.word_embeddings_placeholder = tf.placeholder(tf.float32,[Params.vocab_size, Params.emb_size],"word_embeddings_placeholder")
            self.emb_assign = tf.assign(self.word_embeddings, self.word_embeddings_placeholder)

        # Embed the question and passage information for word and character tokens
        self.passage_word_encoded = encoding(self.passage_w,
                                             word_embeddings = self.word_embeddings,
                                             scope = "passage_embeddings")
        #print(self.passage_word_encoded);print(2)
        self.question_word_encoded = encoding(self.question_w,
                                              word_embeddings = self.word_embeddings,
                                              scope = "question_embeddings")
        #print(self.question_word_encoded);print(4)

        # Passage and question encoding
        #cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _ in range(2)]
        self.passage_encoding = bidirectional_GRU(self.passage_word_encoded,
                                                  self.passage_w_len,
                                                  cell_fn = SRUCell if Params.SRU else GRUCell,
                                                  layers = Params.num_layers,
                                                  scope = "passage_encoding",
                                                  output = 0,
                                                  is_training = self.is_training)
        #print(self.passage_encoding);print(10);
        #cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _ in range(2)]
        self.question_encoding = bidirectional_GRU(self.question_word_encoded,
                                                   self.question_w_len,
                                                   cell_fn = SRUCell if Params.SRU else GRUCell,
                                                   layers = Params.num_layers,
                                                   scope = "question_encoding",
                                                   output = 0,
                                                   is_training = self.is_training)
        #print(self.question_encoding);print(11);

    def attention_match_rnn(self):
        # Apply gated attention recurrent network for both query-passage matching and self matching networks
        with tf.variable_scope("attention_match_rnn"):
            memory = self.question_encoding
            inputs = self.passage_encoding
            scopes = ["question_passage_matching", "self_matching"]
            params = [(([self.params["W_u_Q"],
                         self.params["W_u_P"],
                         self.params["W_v_P"]],self.params["v"]),
                       self.params["W_g"]),
                      (([self.params["W_v_P_2"],
                         self.params["W_v_Phat"]],self.params["v"]),
                       self.params["W_g"])]
            for i in range(2):
                args = {"num_units": Params.attn_size,
                        "memory": memory,
                        "params": params[i],
                        "self_matching": False if i == 0 else True,
                        "memory_len": self.question_w_len if i == 0 else self.passage_w_len,
                        "is_training": self.is_training,
                        "use_SRU": Params.SRU}
                cell = [apply_dropout(gated_attention_Wrapper(**args), size = inputs.shape[-1], is_training = self.is_training) for _ in range(2)]
                inputs = attention_rnn(inputs,
                                       self.passage_w_len,
                                       Params.attn_size,
                                       cell,
                                       scope = scopes[i])
                memory = inputs # self matching (attention over itself)
            self.self_matching_output = inputs

    def bidirectional_readout(self):
        self.final_bidirectional_outputs = bidirectional_GRU(self.self_matching_output,
                                                             self.passage_w_len,
                                                             cell_fn = SRUCell if Params.SRU else GRUCell,
                                                             # layers = Params.num_layers, # or 1? not specified in the original paper
                                                             scope = "bidirectional_readout",
                                                             output = 0,
                                                             is_training = self.is_training)

    def pointer_network(self):
        params = (([self.params["W_u_Q"],self.params["W_v_Q"]],self.params["v"]),
                  ([self.params["W_h_P"],self.params["W_h_a"]],self.params["v"]))
        cell = apply_dropout(SRUCell(Params.attn_size*2), size = self.final_bidirectional_outputs.shape[-1], is_training = self.is_training)
        self.points_logits = pointer_net(self.final_bidirectional_outputs, self.passage_w_len, self.question_encoding, self.question_w_len, cell, params, scope = "pointer_network")

    def outputs(self):
        self.output_index = tf.argmax(self.points_logits, axis = 2)

    def loss_function(self):
        with tf.variable_scope("loss"):
            shapes = self.passage_w.shape
            self.indices_prob = tf.one_hot(self.indices, shapes[1])
            self.mean_loss = cross_entropy(self.points_logits, self.indices_prob)
            self.optimizer = optimizer_factory[Params.optimizer](**Params.opt_arg[Params.optimizer])

            if Params.clip:
                # gradient clipping by norm
                gradients, variables = zip(*self.optimizer.compute_gradients(self.mean_loss))
                gradients, _ = tf.clip_by_global_norm(gradients, Params.norm)
                self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step = self.global_step)
            else:
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step = self.global_step)

    def summary(self):
        self.F1 = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="F1")
        self.F1_placeholder = tf.placeholder(tf.float32, shape = (), name = "F1_placeholder")
        self.EM = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="EM")
        self.EM_placeholder = tf.placeholder(tf.float32, shape = (), name = "EM_placeholder")
        self.valid_loss = tf.Variable(tf.constant(5.0, shape=(), dtype = tf.float32),trainable=False, name="valid_loss")
        self.valid_loss_placeholder = tf.placeholder(tf.float32, shape = (), name = "valid_loss")
        self.metric_assign = tf.group(tf.assign(self.F1, self.F1_placeholder),tf.assign(self.EM, self.EM_placeholder),tf.assign(self.valid_loss, self.valid_loss_placeholder))
        tf.summary.scalar('loss_training', self.mean_loss)
        tf.summary.scalar('loss_valid', self.valid_loss)
        tf.summary.scalar("F1_Score",self.F1)
        tf.summary.scalar("Exact_Match",self.EM)
        tf.summary.scalar('learning_rate', Params.opt_arg[Params.optimizer]['learning_rate'])
        self.merged = tf.summary.merge_all()

def debug():
    model = Model(is_training = True)
    print("Built model")

def test():
    model = Model(is_training = False); print("Built model")
    dict_ = pickle.load(open(Params.input_ + "dictionary.pkl","rb"))
    with model.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(Params.model_))
            q_num = 0
            q_unknown = 0
            q_ids = ['0'] * Params.num_questions
            q_ans = [['0','1']] * Params.num_questions
            for step in tqdm(range(model.num_batch + 1), total = model.num_batch + 1, ncols=70, leave=False, unit='b'):
                if q_num > Params.num_questions: break
                index, _, passage, question = sess.run([model.output_index, model.indices, model.passage_w, model.question_w])####
                q_ind, no_match = find_q(question)
                q_unknown += no_match
                pointers, q_id,aaa = context_pointers(index = index, q_ind = q_ind)####
                for batch in range(Params.batch_size):
                    q_num += 1
                    #print(dict_.ind2word(passage[batch][index[batch][0]:index[batch][1]]))####
                    #print(aaa[batch][pointers[batch][0]:pointers[batch][1]])####
                    #print('-----------------')#####
                    if q_num > Params.num_questions: break
                    if batch >= len(index): break
                    ind_ = q_ind[batch]
                    q_ids[ind_] = q_id[batch]
                    q_ans[ind_] = pointers[batch]
            print('unknown questions: %d' % q_unknown)
            write_predict(q_ids, q_ans)

def main():
    model = Model(is_training = True); print("Built model")
    dict_ = pickle.load(open(Params.input_ + "dictionary.pkl","rb"))
    init = False
    validata, valid_ind = get_valid()
    if not os.path.isfile(os.path.join(Params.model_train,"checkpoint")):
        init = True
        glove = np.memmap(Params.input_ + "glove.np", dtype = np.float32, mode = "r")
        glove = np.reshape(glove,(Params.vocab_size,Params.emb_size))
    with model.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor(logdir=Params.model_train,
                                 save_model_secs=0,
                                 global_step = model.global_step,
                                 init_op = model.init_op)
        with sv.managed_session(config = config) as sess:
            if init: sess.run(model.emb_assign, {model.word_embeddings_placeholder:glove})
            for epoch in range(1, Params.num_epochs+1):
                if sv.should_stop(): break
                for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(model.train_op)
                    if step % Params.save_steps == 0:
                        gs = sess.run(model.global_step)
                        sv.saver.save(sess, Params.model_train + '/model_epoch_%d_step_%d'%(gs//model.num_batch, gs%model.num_batch))
                        sample = np.random.choice(valid_ind, Params.batch_size)
                        feed_dict = {data: validata[i][sample] for i,data in enumerate(model.data)}
                        logits, valid_loss = sess.run([model.points_logits, model.mean_loss], feed_dict = feed_dict)
                        index = np.argmax(logits, axis = 2)
                        F1, EM = 0.0, 0.0
                        for batch in range(Params.batch_size):
                            f1, em = f1_and_EM(index[batch], validata[4][sample][batch], validata[0][sample][batch], dict_)
                            F1 += f1
                            EM += em
                        F1 /= float(Params.batch_size)
                        EM /= float(Params.batch_size)
                        sess.run(model.metric_assign,{model.F1_placeholder: F1, model.EM_placeholder: EM, model.valid_loss_placeholder: valid_loss})
                        print("\nvalid_loss: {}\nvalid_Exact_match: {}\nvalid_F1_score: {}".format(valid_loss,EM,F1))
if __name__ == '__main__':
    if Params.mode.lower() == "debug":
        print("Debugging...")
        debug()
    elif Params.mode.lower() == "test":
        print("Testing on valid set...")
        test()
    elif Params.mode.lower() == "train":
        print("Training...")
        main()
    else:
        print("Invalid mode.")
