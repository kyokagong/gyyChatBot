
import tensorflow as tf
import numpy as np

import sys

from chatbot.src.python.tf_utils import SessionHandler, FLAGS
from tensorflow.python.ops import rnn, rnn_cell
from chatbot.src.python.word_utils import create_vocab_dict, setence2token

import chatbot.src.python.utils.data_util as data_util

class FullForwardLayer():
    def __init__(self, input_size, output_size, weights=None, biases=None):
        self._layer_type = "fc"
        self._input_size = input_size
        self._output_size = output_size

        self._init_weights = weights
        self._init_biases = biases

        self._update_weights_op = None
        self._update_biases_op = None

    def set_init_values(self, weights=None, biases=None):
        self._init_weights = weights
        self._init_biases = biases

    def init_net(self):
        if self._init_weights is None:
            # fully connected, depth 512
            self._init_weights = tf.random_uniform([self._input_size, self._output_size],
                                                minval = -FLAGS.INIT_VAL,
                                                maxval = FLAGS.INIT_VAL)

        if self._init_biases is None:
            self._init_biases = tf.constant(0.0, shape=[self._output_size])

        self._fc_weights = tf.Variable(self._init_weights)
        self._fc_biases = tf.Variable(self._init_biases)

    def get_output(self, input):
        self.input = input
        return tf.matmul(self.input, self._fc_weights) + self._fc_biases


class LSTMLayer():
    def __init__(self,n_hidden):
        self._n_hidden = n_hidden

    def init_net(self):
        with tf.device(FLAGS.device):
            self._init_layer()

    def _init_layer(self):
        self._lstm_cell = rnn_cell.BasicLSTMCell(self._n_hidden, forget_bias=1.0)

    def get_output(self, input):
        self._lstm_cell()
        outputs, states = rnn.rnn(self._lstm_cell, input, dtype=tf.float32)
        return outputs

class RNN():
    def __init__(self, n_input, n_label):
        self._layer_map = {}
        self._layer_num = 0

        self._n_input = n_input
        self._n_label = n_label

        self._input_node = tf.placeholder(dtype=tf.float32, shape=[None, None, n_input])
        self._label_node = tf.placeholder(dtype=tf.float32, shape=[None, n_label])
        self._batch_size = tf.placeholder(dtype=tf.int32)

        self._learning_rate = tf.placeholder(dtype=tf.float32)
        self._gobal_step = tf.Variable(0, trainable=False)
    def add(self, layer):
        self._layer_map[self._layer_num] = layer
        self._layer_num += 1

    def compile_net(self):
        for i in range(self._layer_num):
            self._layer_map[i].init_net()

    def make_net(self):
        self.compile_net()

        next_input = self._layer_map[0].get_output(self._input_node)
        for i in range(1, self._layer_num):
            next_input = self._layer_map[i].get_output(next_input)

        self._logits = next_input
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self._logits, self._label_node))
        self._pred = tf.nn.softmax(self._logits)

        self._optimizer = tf.train.MomentumOptimizer(self._learning_rate, 0.9
                                        ).minimize(self._loss,
                                                        global_step=None)

    def fit(self,inputX, inputY, epoch, batch_size, lr=0.01, decay_size=20, verbose=False):
        lost = 0
        for i in range(epoch):
            feed_dict = {self._input_node: inputX,
                         self._label_node: inputY,
                         self._batch_size: batch_size,
                         self.learning_rate: lr}

            [_, l] = SessionHandler().get_session().run([self.optimizer, self.loss],
                                                        feed_dict=feed_dict)
            lost += l
            if verbose:
                print("loss is " + str(l))
        return lost

class Seq2seq4SameEmbed:
    """
        :arg
           layer_size: the number of unit of lstm cell
    """
    def __init__(self, vocab_size, layer_size, embedding_size,
                     num_layers, batch_size, max_gradient_norm):
        self._vocab_size = vocab_size
        self._buckets = []
        self._layer_size = layer_size
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._batch_size = batch_size
        self._max_gradient_norm = max_gradient_norm
        # self._decay_factor = decay_factor
        # self._dtype = dtype
        self._learning_rate = tf.placeholder(dtype=tf.float32)
        self._global_step = tf.Variable(0, trainable=False)

        self._gradient_norms = []
        self._updates = []

    def add_bucket(self, bucket):
        self._buckets.append(bucket)

    def make_net(self):

        with tf.device(FLAGS.device):
            # use max to pick up the max bucket.
            self.embed_en_inputs = tf.split(0, max(self._buckets, key=lambda x:x[0])[0],
                                            tf.placeholder(dtype=tf.int32,shape=(None,)))
            self.embed_decod_inputs = tf.split(0, max(self._buckets, key=lambda x: x[1])[1],
                                            tf.placeholder(dtype=tf.int32, shape=(None,)))

            # this target_weights is used for weights the target, that means that weights * target to correct the outputs
            # so this target_weights is not the one used in hidden layer to output layer. it's just a 1D TENSOR
            self.target_weights = []
            for i in range(max(self._buckets, key=lambda x: x[1])[1]):
                self.target_weights.append(tf.placeholder(dtype=tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))

            self.targets = [self.embed_decod_inputs[i]
                           for i in range(len(self.embed_decod_inputs))]

            single_cell = tf.nn.rnn_cell.GRUCell(self._layer_size)
            use_lstm = True
            if use_lstm:
                single_cell = tf.nn.rnn_cell.BasicLSTMCell(self._layer_size)
            cell = single_cell
            if self._num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self._num_layers)

            # The seq2seq function: we use embedding for the input.
            def seq2seq_f(encoder_inputs, decoder_inputs):
                return tf.nn.seq2seq.embedding_tied_rnn_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    num_symbols = self._vocab_size,
                    embedding_size = self._embedding_size,
                    output_projection = None,
                    feed_previous=True)

            self._outputs, self._losses = tf.nn.seq2seq.model_with_buckets(
                self.embed_en_inputs, self.embed_decod_inputs, self.targets,
                self.target_weights, self._buckets,
                lambda en,de:seq2seq_f(en,de),
                softmax_loss_function=None)

            _opt = tf.train.MomentumOptimizer(self._learning_rate, 0.9)

            params = tf.trainable_variables()
            for b in range(len(self._buckets)):
                gradients = tf.gradients(self._losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 self._max_gradient_norm)
                self._gradient_norms.append(norm)
                self._updates.append(_opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self._global_step))

            ## create a saver
            self._saver = tf.train.Saver(tf.all_variables())


    def fit(self, train_encode_inputs, train_decode_inputs, target_weights, bucket,
            learning_rate, epoch, verbose=False):
        encoder_size, decoder_size = bucket
        bucket_id = self._buckets.index(bucket)
        output_feed = [self._updates[bucket_id],  # Update Op that does SGD.
                       self._gradient_norms[bucket_id],  # Gradient norm.
                       self._losses[bucket_id]]  # Loss for this batch.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.embed_en_inputs[l].name] = train_encode_inputs[l]
        for l in range(decoder_size):
            input_feed[self.embed_decod_inputs[l].name] = train_decode_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        input_feed[self._learning_rate] = learning_rate
        for i in range(epoch):
            _, gradients, loss = SessionHandler().get_session().run(output_feed, feed_dict=input_feed)
            if verbose:
                print("epoch %s, loss is: %s" % (i, loss))
        return loss


    def predict(self, encode_inputs, target_weights, bucket):
        # Get output logits for the sentence.
        # print(encode_inputs[0])
        # print(type(encode_inputs[0]))
        bucket_id = self._buckets.index(bucket)

        input_feed = {}
        for i in range(bucket[0]):
            input_feed[self.embed_en_inputs[i].name] = encode_inputs[i]
        for i in range(bucket[1]):
            input_feed[self.embed_decod_inputs[i].name] = np.zeros(encode_inputs[0].shape[0], dtype=np.int32)
            input_feed[self.target_weights[i].name] = np.zeros(encode_inputs[0].shape[0], dtype=np.float32)

        outputs = SessionHandler().get_session().run(self._outputs[bucket_id], feed_dict=input_feed)

        # in python3, argmax outputs integer value
        outputs_token_id = [np.argmax(logit, axis=1) for logit in outputs]
        return outputs_token_id

    def save_weights_variables(self, checkpoint_path):
        self._saver.save(SessionHandler().get_session(), checkpoint_path, global_step=self._global_step)

    def restore_weights_variables(self, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self._saver.restore(SessionHandler().get_session(), ckpt.model_checkpoint_path)


def test_data():
    digit = np.loadtxt("../../static/digitInputOutput.txt", dtype=np.float32)
    np.random.shuffle(digit)
    train_size = 4000
    train_X = digit[:train_size, 0:400]
    train_Y = digit[:train_size, 400:]

    test_X = digit[train_size:, 0:400]
    test_Y = digit[train_size:, 400:]
    return (train_X, train_Y, test_X, test_Y)

def test_tokens():
    en_data = open("../../../test/sample_data_en.txt", "r").readlines()
    fr_data = open("../../../test/sample_data_fr.txt", "r").readlines()
    vocab_dict_en, vocab_dict_inv_en = create_vocab_dict("../../../test/vocabulary_en.txt")
    vocab_dict_fr, vocab_dict_inv_fr = create_vocab_dict("../../../test/vocabulary_fr.txt")
    en_token_list = [setence2token(sentence.strip(), vocab_dict_inv_en) for sentence in en_data]
    fr_token_list = [setence2token(sentence.strip(), vocab_dict_inv_fr) for sentence in fr_data]
    a = 1

def test_seq2seq():
    pass


def test():
    with tf.device(FLAGS.device):
        size = tf.placeholder(dtype=tf.int32)
        x_node = tf.placeholder(dtype=tf.float32)
        x_split = tf.split(0, 10, x_node)
        with tf.Session() as sess:
            ran = np.random.random((11,10))
            print()
            output = x_split[0].eval(feed_dict={
                                             x_node:ran})
            print(output)


if __name__ == '__main__':
    # SessionHandler().set_default()
    # test_lstm()
    test_seq2seq()