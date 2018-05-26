import random

import numpy as np
import tensorflow as tf

import data_util

emb_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
fc_layer = tf.contrib.layers.fully_connected

class Hidden_Classifier(object):
    def __init__(self,
                source_vocab_size,
                target_vocab_size,
                buckets,
                state_size,
                num_layers,
                embedding_size,
                max_gradient,
                batch_size,
                learning_rate,
                forward_only=False,
                dtype=tf.float32):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.output_size = buckets[0][0]
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.state_size = state_size
        self.diff_inputs = tf.placeholder(
            tf.int32, shape=[None, buckets[0][0]])
        self.lables = tf.placeholder(
            tf.float32, shape=[None, buckets[0][0]])
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.lables_len = tf.placeholder(tf.int32, shape=[None])

        #cell for hidden classification
        cell1 = tf.contrib.rnn.GRUCell(state_size)
        cell2 = tf.contrib.rnn.GRUCell(state_size)

        if not forward_only:
            cell1 = tf.contrib.rnn.DropoutWrapper(
                cell1, output_keep_prob = 0.8)
            cell2 = tf.contrib.rnn.DropoutWrapper(
                cell2, output_keep_prob = 0.8)

        encoder_emb = tf.get_variable(
            "embedding", [source_vocab_size, embedding_size],
            initializer=emb_init)

        encoder_inputs_emb = tf.nn.embedding_lookup(
            encoder_emb, self.diff_inputs)

           
        with tf.variable_scope("hid_classification"):
            multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
            outputs, state = tf.nn.dynamic_rnn(multi_cell, encoder_inputs_emb, dtype = tf.float32)
            outputs = tf.transpose (outputs, [1,0,2])
            outputs = outputs[-1]
                
            self.output_sigmoid = fc_layer(outputs,self.output_size, tf.sigmoid)

            self.loss = tf.reduce_mean(-(self.lables* tf.log(self.output_sigmoid))-(1-self.lables)*tf.log(1-self.output_sigmoid))
            lr = tf.train.exponential_decay(0.001, self.global_step, 10000, 0.75)
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss, self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
        self.summary_merge = tf.summary.merge_all()

    def step(self,
             session,
             encoder_inputs,
             lables, 
             encoder_len,
             forward_only,
             summary_writer=None):

        # dim fit is important for sequence_mask
        # TODO better way to use sequence_mask
        input_feed = {}
        input_feed[self.diff_inputs] = encoder_inputs
        input_feed[self.lables]= lables
        input_feed[self.lables_len] = encoder_len
        if forward_only:
            output_feed = [self.loss, self.output_sigmoid]
        else:
            output_feed = [self.train_step, self.loss, self.output_sigmoid]


        outputs = session.run(output_feed, input_feed)
        #print(outputs[0],outputs[1])
        return outputs