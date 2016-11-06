
import re

def setence2token(sentence, vocab_dict_inv):
    """
    :param sentence: a sentence can be splitted by " "
    :param vocab_dict_inv:
    :return:
    """
    token_ids_list = [ vocab_dict_inv[word] for word in re.split(r'\s*', sentence.strip())]
    return token_ids_list



def create_vocab_dict(vocab_file, base_token_id):
    """
    :param vocab_file:
    :return:  a token_index, word dictionary
    """
    vocab_dict = {}
    vocab_dict_inv = {}
    with open(vocab_file, 'r') as f:
        line = f.readline()
        line_num = base_token_id
        while line:
            vocab_dict[line_num] = line.strip()
            vocab_dict_inv[line.strip()] = line_num
            line_num += 1
            line = f.readline()
    return (vocab_dict, vocab_dict_inv)
