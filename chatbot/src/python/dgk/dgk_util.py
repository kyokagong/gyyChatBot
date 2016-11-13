
import sys
import os

# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import time

from chatbot.src.python.word_utils import create_vocab_dict
from chatbot.src.python.nn_utils import Seq2seq4SameEmbed
from chatbot.src.python.tf_utils import SessionHandler
import chatbot.src.python.utils.data_util as data_util


DIR = "../../../../test/%s"
# DIR = "/Users/kyoka/GitHub/gyyChatBot/test/%s"
DGK_DATA = DIR%"dgk_data3.txt"
BATCH_SIZE = 128

DGK_BUCKETS = [(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70)]

def create_vocab():
    vocab_dict = {}
    with open(DGK_DATA,'r') as f:
        line = f.readline()
        while line:
            line = line.strip().split(" ")
            if len(line) >= 2:
                line = line[1].split('/')
                for item in line:
                    try:
                        vocab_dict[item] += 1
                    except:
                        vocab_dict[item] = 1
            line = f.readline()

    with open(DIR%"dgk_vocabulary.txt",'w') as output:
        for item in vocab_dict:
            output.write(item+"\n")

def get_pair_sample(vocab_dict_inv, sample_size=10000000):
    pair_data = []
    with open(DGK_DATA,'r') as f:
        line = f.readline()
        line_num = 0
        while line:
            line = line.strip().split(" ")
            if line[0] == "E":
                last_ask = None
            elif line[0] == "M" and len(line) >= 2:
                items = line[1]
                answer = [vocab_dict_inv[word] for word in items.split("/")]
                if last_ask is not None:
                    pair_data.append((last_ask, answer))
                last_ask = answer
            line = f.readline()
            line_num += 1
            if line_num >= sample_size:
                break
    return pair_data

def get_batch_buckets_map(pair_data, buckets):
    """
        first constructing a map contain <bucket, list of pairs>
        second, random pick a bucket, and train pairs in buckets_map[bucket] batch by batch

        args: buckets must be order in asc
    """
    buckets_map = {}
    for bucket in list(reversed(buckets)):
        buckets_map[bucket] = []

    for pair in pair_data:
        for bucket in buckets:
            if max([len(pair[0]),len(pair[1])]) <= min(bucket)-3:
                buckets_map[bucket].append(pair)
                break
    return buckets_map

def get_batch_input_pairs(buckets_map, bucket, batch_size):
    """
        an iterator for creating encode_inputs and decoder_inputs
    :param buckets_map:
    :param bucket:
    :param batch_size:
    :return:
    """
    encoder_size, decoder_size = bucket
    num_batch = int(math.floor(len(buckets_map[bucket])/batch_size))
    if num_batch == 0:
        pairs = buckets_map[bucket]
        yield _get_encode_decode_inputs(pairs, encoder_size, decoder_size, len(pairs))
    else:
        for i in range(num_batch):
            batch_index = i * batch_size
            pairs = buckets_map[bucket][batch_index:(batch_index+batch_size)]
            yield _get_encode_decode_inputs(pairs, encoder_size, decoder_size, batch_size)
        # the rest of the data_pairs
        rest_pairs = buckets_map[bucket][(batch_index+batch_size):]
        if len(rest_pairs) > 0:
            yield _get_encode_decode_inputs(rest_pairs, encoder_size, decoder_size, len(rest_pairs))


def _get_encode_decode_inputs(pairs, encoder_size, decoder_size, batch_size):
    encoder_inputs, decoder_inputs = [], []
    for pair in pairs:
        encoder_pad = [data_util.PAD_ID] * (encoder_size - len(pair[0]))
        decoder_pad = [data_util.PAD_ID] * (decoder_size - len(pair[1]))

        encoder_inputs.append(encoder_pad + list(pair[0]))
        decoder_inputs.append([data_util.GO_ID] + pair[1] + decoder_pad)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    for length_idx in range(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in range(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)], dtype=np.int32))

        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in range(batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_util.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)

    return (batch_encoder_inputs, batch_decoder_inputs, batch_weights)


def write_pair_line(ask_list, answer_list):
    left = ",".join([str(item) for item in ask_list])
    right = ",".join([str(item) for item in answer_list])
    with open(DIR%"dgk_pair_token_ids.txt",'a') as output:
        output.write(left+"\t"+right)





def create_seq2seq_model(vocab_size, layer_size, embedding_size,
                     num_layers, batch_size, max_gradient_norm, buckets):
    seq2seq = Seq2seq4SameEmbed(vocab_size, layer_size, embedding_size,
                     num_layers, batch_size, max_gradient_norm)
    for bucket in buckets:
        seq2seq.add_bucket(bucket)

    seq2seq.make_net()
    return seq2seq


def dgk_main():
    # SessionHandler().set_default()

    vocab_dict, vocab_dict_inv = create_vocab_dict(DIR%"dgk_vocabulary.txt", 3)
    pair_sample = get_pair_sample(vocab_dict_inv)
    print(len(pair_sample))
    print(sys.getsizeof(pair_sample))

    buckets_map = get_batch_buckets_map(pair_sample, DGK_BUCKETS)

    sum_pairs = 0
    for item in buckets_map:
        sum_pairs += len(buckets_map[item])
        print(item, len(buckets_map[item]))
    print(sum_pairs)

    seq2seq = create_seq2seq_model(len(vocab_dict) + 3, 20, 20, 1, BATCH_SIZE, 5.0, DGK_BUCKETS)


    # SessionHandler().initialize_variables()



    epoches = 10
    run_count = 0
    base = 100
    bucket_id = 0

    with tf.Session() as sess:
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)
        v_list = tf.all_variables()
        for v in v_list:
            if not tf.is_variable_initialized(v).eval():
                print(v)
                SessionHandler().get_session().run(v.initialized_value())


        print("start dgk run")

        seq2seq.restore_weights_variables(sess, DIR % "ckpt/")

        start_time = time.time()
        for epoch in range(epoches):
            # for bucket_id in range(len(DGK_BUCKETS)):

                if len(buckets_map[DGK_BUCKETS[bucket_id]]) > 0:
                    for encoder_inputs, decoder_inputs, target_weights in \
                            get_batch_input_pairs(buckets_map, DGK_BUCKETS[bucket_id], BATCH_SIZE):
                        loss = seq2seq.fit(sess, encoder_inputs, decoder_inputs, target_weights, DGK_BUCKETS[bucket_id], 0.01, 1, True)
                        # pred_tokens = seq2seq.predict(encoder_inputs, target_weights, DGK_BUCKETS[bucket_id])
                        # print(pred_tokens)
                        log = "time:%s epoch:%s run count:%s loss:%s" % (
                            time.asctime(time.localtime(time.time())),
                            epoch,
                            run_count,
                            loss)
                        # if run_count % base == 0:
                            # write_dgk_log(log+"\n")

                        print(log)
                        run_count += 1

        seq2seq.save_weights_variables(DIR%"ckpt/translate.ckpt")

        end_time = time.time()
        print(end_time-start_time)

    # len_en_list = []
    # len_de_list = []
    # for pair in pair_sample:
    #     len_en_list.append(len(pair[0]))
    #     len_de_list.append(len(pair[1]))
    #
    # print("encode: max len:%s, min len:%s" % (max(len_en_list), min(len_en_list)))
    # print("decoder: max len:%s, min len:%s" % (max(len_de_list), min(len_de_list)))
    #
    # plt.hist(np.array(len_en_list), 50, normed=1, facecolor='g', alpha=0.75)
    # plt.show()


    """clear error data """
    # with open("dgk_data3.txt",'r') as f:
    #     # with open("dgk_data3.txt",'w') as output:
    #         for line in f.readlines():
    #             if (len(line.strip().split(" ")) >= 2 and \
    #                     len(line.strip().split(" ")[1].split("/")) >= 100):
    #                 # output.write(line)
    #                 print(line.strip())

def write_dgk_log(output_log):
    data = None
    output_data = [output_log]
    try:
        with open(data_util.LOG_FILE, "r") as output:
            data = output.readlines()
    except:
        pass
    with open(data_util.LOG_FILE, "w") as output:
        if data is not None:
            for i in range(len(data), 0, -1):
                output_data.append(data[i - 1])
        output.writelines(output_data)

def test_main():
    print(sys.path[0])
    dgk_main()

def sleep_task(t=10):
    print("start sleep")
    time.sleep(t)


if __name__ == '__main__':
    test_main()

    # open(DIR%"dgk_vocabulary.txt",'r')
    # print(sys.path[0])