import numpy as np
import tensorflow as tf

class hidClass(object):
    def __init__(self, iteration, batch_size, learning_rate, hidden_size):
        self.iteration = iteration
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.output_size = 1

        self.hidden_input = tf.placeholder(tf.float32, shape = [self.batch_size, None, hidden_size*2])
        self.hidden_label = tf.placeholder(tf.int32, shape = [self.batch_size, None])
        self.dorate = tf.placeholder(tf.float32)

        cell1 = tf.contrib.rnn.GRUCell(self.hidden_size)
        cell1 = tf.contrib.rnn.DropoutWrapper(cell1, output_keep_prob = self.dorate)
        cell2 = tf.contrib.rnn.GRUCell(self.hidden_size)
        cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob = self.dorate)
        
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
        outputs, state = tf.nn.dynamic_rnn(multi_cell, embedded, dtype = tf.float32)
        outputs = tf.transpose (outputs, [1,0,2])
        outputs = outputs[-1]

        W = tf.Variable(tf.truncated_normal([self.hidden_size, self.output_size]))
        b = tf.Variable(tf.constant(0.1, [self.output]))
        output = tf.matmul(outputs, W) + b
        output_sigmoid = tf.sigmoid(output)

        loss = tf.reduce_mean(-(hidden_label* tf.log(output_sigmoid))-(1-hidden_label)*tf.log(1-output_sigmoid))
        global_step = tf.Variable(0)
        lr = tf.train.exponential_decay(self.learning_rate, global_step, 10000, 0.75)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step)

    def step (self,session, hidden_input, hidden_output, dropout_rate, forward_only):
        feed_inp = {}
        feed_inp[hidden_input] = hidden_input
        feed_inp[hidden_label] = hidden_output
        feed_inp[dorate] = dropout_rate

        if(forward_only):
                output_feed = [loss, output_sigmoid]
        else:
                output_feed = [loss, train_step]

        outputs = session.run(output_feed, input_feed)

        return outputs

