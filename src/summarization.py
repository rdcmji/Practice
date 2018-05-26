import logging
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

import bigru_model
import hidd_classifier
import data_util
from sklearn.metrics import confusion_matrix

tf.app.flags.DEFINE_float("learning_rate", 1., "Learning rate.")
tf.app.flags.DEFINE_integer("size", 150, "Size of hidden layers.")
tf.app.flags.DEFINE_integer("embsize", 150, "Size of embedding.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("test_file", 'data/test.1981.diff.txt', "Test filename.")
tf.app.flags.DEFINE_string("test_output", "output.txt", "Test output.")
tf.app.flags.DEFINE_string("train_dir", "model", "Training directory.")
tf.app.flags.DEFINE_string("class_train_dir", "class_model", "Classification training directory.")
tf.app.flags.DEFINE_string("tfboard", "tfboard", "Tensorboard log directory.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for testing.")
tf.app.flags.DEFINE_boolean("geneos", True, "Do not generate EOS. ")
tf.app.flags.DEFINE_float(
    "max_gradient", 1.0, "Clip gradients l2 norm to this range.")
tf.app.flags.DEFINE_integer(
    "batch_size", 160, "Batch size in training / beam size in testing.")
tf.app.flags.DEFINE_integer(
    "doc_vocab_size", 5000, "Document vocabulary size.")
tf.app.flags.DEFINE_integer(
    "sum_vocab_size", 1000, "Summary vocabulary size.")
tf.app.flags.DEFINE_integer(
    "max_train", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer(
    "max_iter", 1000, "Maximum training iterations.")
tf.app.flags.DEFINE_integer(
    "class_max_iter", 1000, "Maximum classification training iterations.")
tf.app.flags.DEFINE_integer(
    "steps_per_validation", 1000, "Training steps between validations.")
tf.app.flags.DEFINE_integer(
    "steps_per_checkpoint", 1000, "Training steps between checkpoints.")
tf.app.flags.DEFINE_string(
    "checkpoint", "", "Checkpoint to load (use up-to-date if not set)")
tf.app.flags.DEFINE_string(
    "cl_checkpoint", "", "Classification Checkpoint to load (use up-to-date if not set)")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets for sampling
#_buckets = [(60,10),(80,10),(90,10),(80, 30), (120,30)]
_buckets = [(100, 30, 30)]


def create_bucket(source, target, hidd_target):
    data_set = [[] for _ in _buckets]
    for s, t, ht in zip(source, target, hidd_target):
        t = [data_util.ID_GO] + t + [data_util.ID_EOS]
        for bucket_id, (s_size, t_size, ht_size) in enumerate(_buckets):
            if len(s) <= s_size and len(t) <= t_size:
                data_set[bucket_id].append([s, t, ht])
                break
    return data_set

#def getAcc(lable, pred):
    


def create_model(session, forward_only):
    """Create model and initialize or load parameters in session."""
    dtype = tf.float32
    model = bigru_model.BiGRUModel(
        FLAGS.doc_vocab_size,
        FLAGS.sum_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.embsize,
        FLAGS.max_gradient,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        forward_only=forward_only,
        dtype=dtype)
    if FLAGS.checkpoint != "":
        ckpt = FLAGS.checkpoint
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt:
            ckpt = ckpt.model_checkpoint_path
    if ckpt and tf.train.checkpoint_exists(ckpt):
        logging.info("Reading model parameters from %s" % ckpt)
        model.saver.restore(session, ckpt)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def create_class_model(session, forward_only):
    dtype = tf.float32
    model = hidd_classifier.Hidden_Classifier(
        FLAGS.doc_vocab_size,
        FLAGS.sum_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.embsize,
        FLAGS.max_gradient,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        forward_only=forward_only,
        dtype=dtype)
    if FLAGS.cl_checkpoint != "":
        ckpt = FLAGS.cl_checkpoint
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.class_train_dir)
        if ckpt:
            ckpt = ckpt.model_checkpoint_path
    if ckpt and tf.train.checkpoint_exists(ckpt):
        logging.info("Reading model parameters from %s" % ckpt)
        model.saver.restore(session, ckpt)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    logging.info("Preparing summarization data.")
    docid, sumid, doc_dict, sum_dict, hidden_label= \
        data_util.load_data(
            FLAGS.data_dir + "/train.48615.diff",
            FLAGS.data_dir + "/train.48615.msg",
            FLAGS.data_dir + "/doc_dict.txt",
            FLAGS.data_dir + "/sum_dict.txt",
            FLAGS.doc_vocab_size, FLAGS.sum_vocab_size)

    val_docid, val_sumid, val_hidd_label = \
        data_util.load_valid_data(
            FLAGS.data_dir + "/valid.3000.diff",
            FLAGS.data_dir + "/valid.3000.msg",
            doc_dict, sum_dict)

    

    with tf.Session() as sess:
        # Create model.
        logging.info("Creating %d layers of %d units." %
                     (FLAGS.num_layers, FLAGS.size))
        train_writer = tf.summary.FileWriter(FLAGS.tfboard, sess.graph)
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        logging.info("Create buckets.")
        dev_set = create_bucket(val_docid, val_sumid, val_hidd_label)
        train_set = create_bucket(docid, sumid, hidden_label)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [
            sum(train_bucket_sizes[:i + 1]) / train_total_size
            for i in range(len(train_bucket_sizes))]

        for (s_size, t_size, _), nsample in zip(_buckets, train_bucket_sizes):
            logging.info("Train set bucket ({}, {}) has {} samples.".format(
                s_size, t_size, nsample))

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        
        current_step = sess.run(model.global_step)

        while current_step < FLAGS.max_iter:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, encoder_len, decoder_len, class_output, class_len = \
                data_util.get_batch(train_set, _buckets, bucket_id, FLAGS.batch_size, False,0)
            step_loss, _ = model.step(
                sess, encoder_inputs, decoder_inputs,
                encoder_len, decoder_len, False, train_writer)

            step_time += (time.time() - start_time) / \
                FLAGS.steps_per_validation
            loss += step_loss * FLAGS.batch_size / np.sum(decoder_len) \
                / FLAGS.steps_per_validation
            current_step += 1

            # Once in a while, we save checkpoint.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)

            # Once in a while, we print statistics and run evals.
            if current_step % FLAGS.steps_per_validation == 0:
                # Print statistics for the previous epoch.
                perplexity = np.exp(float(loss))
                logging.info(
                    "global step %d step-time %.2f ppl %.2f" % (model.global_step.eval(), step_time, perplexity))

                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        logging.info("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs,  encoder_len, decoder_len, class_output, class_len  =\
                        data_util.get_batch(dev_set, _buckets, bucket_id, FLAGS.batch_size, False, 0)
                    #cl_eval_loss, _ = class_model.step(sess, class_input, class_output, class_len, True)
                    eval_loss, _, _ = model.step(sess, encoder_inputs,
                                            decoder_inputs, encoder_len,
                                            decoder_len, True)
                    eval_loss = eval_loss * FLAGS.batch_size \
                        / np.sum(decoder_len)
                    eval_ppx = np.exp(float(eval_loss))
                    logging.info("  eval: bucket %d ppl %.2f" %
                                 (bucket_id, eval_ppx))
                sys.stdout.flush()

        #Get Encoder outputs
        batchidx=0
        final_inputs= []
        final_outputs = []
        final_len = []
        while batchidx+FLAGS.batch_size <=train_bucket_sizes[0]:
            encoder_inputs, decoder_inputs, encoder_len, decoder_len, class_output, class_len = \
                data_util.get_batch(train_set, _buckets, bucket_id, FLAGS.batch_size, True, batchidx)

            _, _, enc_outputs = model.step(
                sess, encoder_inputs, decoder_inputs,
                encoder_len, decoder_len, True)

            
            enc_outputs = data_util.add_pad_for_hidden(enc_outputs, _buckets[0][0])

            final_inputs.append(enc_outputs)
            final_outputs.append(class_output)
            final_len.append(class_len)

            batchidx+=FLAGS.batch_size

        final_inputs = np.asarray(final_inputs)
        final_inputs = np.concatenate(final_inputs, 0)
        final_outputs = np.asarray(final_outputs)
        final_outputs = np.concatenate(final_outputs, 0)
        final_len = np.asarray(final_len)
        final_len =np.concatenate(final_len, 0)
        print(final_inputs.shape,final_outputs.shape, final_len.shape)

        #Hidden classifier
        class_model = create_class_model (sess, False)
        classification_curr_step = sess.run(class_model.global_step)
        i=0
        while classification_curr_step <= FLAGS.class_max_iter:
            
            _, step_loss, output = class_model.step(
                sess, final_inputs[i:(i+160)], final_outputs[i:(i+160)],
                final_len[i:(i+160)], False)
            classification_curr_step+=1
            
            clipped = np.array(output > 0.5, dtype=np.int)
            #print("i", i)
            #print("clfcurrstep",classification_curr_step)
            #print("clipped", clipped.flatten())
            #print("final_outputs", final_outputs[i:(i+160)].flatten())
            tn, fp, fn, tp = confusion_matrix(final_outputs[i:(i+160)].flatten(), clipped.flatten()).ravel()
            if(classification_curr_step%40 == 0):
                print("Train Precision", tp/(tp+fp+0.1))
                print("Train Accuracy", (tp+tn)/(tp+fp+tn+fn))
            if(i+160 == len(final_len)):
                i = 0
            else:
                i+=160

            # Once in a while, we save checkpoint.
            if classification_curr_step % FLAGS.steps_per_checkpoint == 0:
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.class_train_dir, "class_model.ckpt")
                class_model.saver.save(sess, checkpoint_path,
                                 global_step=class_model.global_step)
        
        print("test_file", FLAGS.test_file)    
        docs, data = data_util.load_test_data(FLAGS.test_file, doc_dict)

        #test

        # Create model and load parameters.
        '''
        logging.info("Creating %d layers of %d units." %
                     (FLAGS.num_layers, FLAGS.size))

        result = []
        for idx, token_ids in enumerate(data):

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, encoder_len, decoder_len, class_output, class_len =\
                data_util.get_batch(
                    {0: [(token_ids, [data_util.ID_GO, data_util.ID_EOS],[0,0])]}, _buckets, 0, FLAGS.batch_size, False, 0)

            if FLAGS.batch_size == 1 and FLAGS.geneos:
                loss, outputs = model.step(sess,
                    encoder_inputs, decoder_inputs,
                    encoder_len, decoder_len, False)

                outputs = [np.argmax(item) for item in outputs[0]]
            else:
                outputs = model.step_beam(
                    sess, encoder_inputs, encoder_len, geneos=FLAGS.geneos)

            # If there is an EOS symbol in outputs, cut them at that point.
            if data_util.ID_EOS in outputs:
                outputs = outputs[:outputs.index(data_util.ID_EOS)]
            gen_sum = " ".join(data_util.sen_map2tok(outputs, sum_dict[1]))
            gen_sum = data_util.sen_postprocess(gen_sum)
            result.append(gen_sum)
            logging.info("Finish {} samples. :: {}".format(idx, gen_sum[:75]))
        '''
            
        #Get Encoder outputs
        docid, sumid, doc_dict, sum_dict, hidden_label= \
        data_util.load_data(
            FLAGS.data_dir + "/test.1981.diff.txt",
            FLAGS.data_dir + "/test.1981.msg.txt",
            FLAGS.data_dir + "/doc_dict.txt",
            FLAGS.data_dir + "/sum_dict.txt",
            FLAGS.doc_vocab_size, FLAGS.sum_vocab_size)
        test_set = create_bucket(docid, sumid, hidden_label)
        test_bucket_sizes = [len(test_set[b]) for b in range(len(_buckets))]
        test_total_size = float(sum(test_bucket_sizes))
        test_buckets_scale = [
            sum(test_bucket_sizes[:i + 1]) / test_total_size
            for i in range(len(test_bucket_sizes))]
        batchidx=0
        final_inputs= []
        final_outputs = []
        final_len = []
        #data.shape == (1, 158, 3) so I changed FLAGS.batch_size
        FLAGS.batch_size = 158
        while batchidx+FLAGS.batch_size <=len(data):
            #bucket_id = (i for i in range(len(test_buckets_scale))
            encoder_inputs, decoder_inputs, encoder_len, decoder_len, class_output, class_len = \
                data_util.get_batch(test_set, _buckets, bucket_id, FLAGS.batch_size, True, batchidx)

            _, _, enc_outputs = model.step(
                sess, encoder_inputs, decoder_inputs,
                encoder_len, decoder_len, True)

            
            enc_outputs = data_util.add_pad_for_hidden(enc_outputs, _buckets[0][0])

            final_inputs.append(enc_outputs)
            final_outputs.append(class_output)
            final_len.append(class_len)

            batchidx+=1

        final_inputs = np.asarray(final_inputs)
        final_inputs = np.concatenate(final_inputs, 0)
        final_outputs = np.asarray(final_outputs)
        final_outputs = np.concatenate(final_outputs, 0)
        final_len = np.asarray(final_len)
        final_len =np.concatenate(final_len, 0)
        print(final_inputs.shape,final_outputs.shape, final_len.shape)

        #Hidden classifier
        step_loss, output = class_model.step(
            sess, final_inputs[:], final_outputs[:],
            final_len[:], True)
        
        clipped = np.array(output > 0.5, dtype=np.int)
        tn, fp, fn, tp = confusion_matrix(final_outputs[:].flatten(), clipped.flatten()).ravel()
        #with open('data/test.1981.msg.txt')as reader:
        #    testmsg=[]
        #    for i in range(1981):
        #        testmsg.append(reader.readline())
        #sums  = list(map(lambda x: x.split(), testmsg))
        #labels = data_util.hidden_label_gen(FLAGS.test_file, sums)
        #tn, fp, fn, tp = confusion_matrix(labels.flatten(), clipped.flatten())
        print("Test Precision : ", tp/(tp+fp+0.1))
        print("Test Accuracy", (tp+tn)/(tp+fp+tn+fn))
        with open(FLAGS.test_output, "w") as f:
            for idx in range(1981):
                for j in range(len(docs[idx])):
                    if clipped[idx][j]==1:
                        print("Recommended identifier: "+docs[idx][j]+" ", file=f)
                print("\n", file=f)


def decode():
    # Load vocabularies.
    doc_dict = data_util.load_dict(FLAGS.data_dir + "/doc_dict.txt")
    sum_dict = data_util.load_dict(FLAGS.data_dir + "/sum_dict.txt")
    if doc_dict is None or sum_dict is None:
        logging.warning("Dict not found.")
    docs, data = data_util.load_test_data(FLAGS.test_file, doc_dict)

    with tf.Session() as sess:
        # Create model and load parameters.
        logging.info("Creating %d layers of %d units." %
                     (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, True)
        class_model = create_class_model(sess, True)

        result = []
        for idx, token_ids in enumerate(data):

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, encoder_len, decoder_len, class_output, class_len =\
                data_util.get_batch(
                    {0: [(token_ids, [data_util.ID_GO, data_util.ID_EOS],[0,0])]}, _buckets, 0, FLAGS.batch_size, False, 0)

            if FLAGS.batch_size == 1 and FLAGS.geneos:
                loss, outputs = model.step(sess,
                    encoder_inputs, decoder_inputs,
                    encoder_len, decoder_len, True)

                outputs = [np.argmax(item) for item in outputs[0]]
            else:
                outputs = model.step_beam(
                    sess, encoder_inputs, encoder_len, geneos=FLAGS.geneos)

            # If there is an EOS symbol in outputs, cut them at that point.
            if data_util.ID_EOS in outputs:
                outputs = outputs[:outputs.index(data_util.ID_EOS)]
            gen_sum = " ".join(data_util.sen_map2tok(outputs, sum_dict[1]))
            gen_sum = data_util.sen_postprocess(gen_sum)
            result.append(gen_sum)
            logging.info("Finish {} samples. :: {}".format(idx, gen_sum[:75]))
            
        #Get Encoder outputs
        batchidx=0
        final_inputs= []
        final_outputs = []
        final_len = []
        while batchidx+FLAGS.batch_size <=len(data):
            encoder_inputs, decoder_inputs, encoder_len, decoder_len, class_output, class_len =\
            data_util.get_batch(
                {0: [(token_ids, [data_util.ID_GO, data_util.ID_EOS],[0,0])]}, _buckets, 0, FLAGS.batch_size, False, 0)

            _, _, enc_outputs = model.step(
                sess, encoder_inputs, decoder_inputs,
                encoder_len, decoder_len, True)

            
            enc_outputs = data_util.add_pad_for_hidden(enc_outputs, _buckets[0][0])

            final_inputs.append(enc_outputs)
            final_outputs.append(class_output)
            final_len.append(class_len)

            batchidx+=FLAGS.batch_size

        final_inputs = np.asarray(final_inputs)
        final_inputs = np.concatenate(final_inputs, 0)
        final_outputs = np.asarray(final_outputs)
        final_outputs = np.concatenate(final_outputs, 0)
        final_len = np.asarray(final_len)
        final_len =np.concatenate(final_len, 0)
        print(final_inputs.shape,final_outputs.shape, final_len.shape)

        #Hidden classifier
        step_loss, output = class_model.step(
            sess, final_inputs[:], final_outputs[:],
            final_len[:], True)
        
        clipped = np.array(output > 0.5, dtype=np.int)
        #label = data_util.hidden_label_gen(FLAGS.test_file, "data/test.1981.msg.txt")
        #make confusion matrix to get precision
        #tn, fp, fn, tp = confusion_matrix(label.flatten(), clipped.flatten()).ravel()
        #print("Test precision : ", tp/(tp+fp))


        with open(FLAGS.test_output, "w") as f:
            for idx, item in enumerate(result):
                print(item, file=f)
                for j in range(len(docs[idx])):
                    if clipped[idx][j]==1:
                        print("Recommended identifier: "+docs[idx][j]+" ", file=f)
                print("\n", file=f)

def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')
    try:
        os.makedirs(FLAGS.train_dir)
    except:
        pass
    tf.app.run()
