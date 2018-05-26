import logging
import random


import numpy as np
import re

MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3


def load_dict(dict_path, max_vocab=None):
    logging.info("Try load dict from {}.".format(dict_path))
    try:
        dict_file = open(dict_path)
        dict_data = dict_file.readlines()
        dict_file.close()
    except:
        logging.info(
            "Load dict {dict} failed, create later.".format(dict=dict_path))
        return None

    dict_data = list(map(lambda x: x.split(), dict_data))
    if max_vocab:
        dict_data = list(filter(lambda x: int(x[0]) < max_vocab, dict_data))
    tok2id = dict(map(lambda x: (x[1], int(x[0])), dict_data))
    id2tok = dict(map(lambda x: (int(x[0]), x[1]), dict_data))
    logging.info(
        "Load dict {} with {} words.".format(dict_path, len(tok2id)))
    return (tok2id, id2tok)


def create_dict(dict_path, corpus, max_vocab=None):
    logging.info("Create dict {}.".format(dict_path))
    counter = {}
    for line in corpus:
        for word in line:
            try:
                counter[word] += 1
            except:
                counter[word] = 1

    for mark_t in MARKS:
        if mark_t in counter:
            del counter[mark_t]
            logging.warning("{} appears in corpus.".format(mark_t))

    counter = list(counter.items())
    counter.sort(key=lambda x: -x[1])
    words = list(map(lambda x: x[0], counter))
    words = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO] + words
    if max_vocab:
        words = words[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    with open(dict_path, 'w') as dict_file:
        for idx, tok in enumerate(words):
            print(idx, tok, file=dict_file)
            tok2id[tok] = idx
            id2tok[idx] = tok

    logging.info(
        "Create dict {} with {} words.".format(dict_path, len(words)))
    return (tok2id, id2tok)


def corpus_map2id(data, tok2id):
    ret = []
    unk = 0
    tot = 0
    for doc in data:
        tmp = []
        for word in doc:
            tot += 1
            try:
                tmp.append(tok2id[word])
            except:
                tmp.append(ID_UNK)
                unk += 1
        ret.append(tmp)
    return ret, (tot - unk) / tot


def sen_map2tok(sen, id2tok):
    return list(map(lambda x: id2tok[x], sen))


def load_data(doc_filename,
              sum_filename,
              doc_dict_path,
              sum_dict_path,
              max_doc_vocab=None,
              max_sum_vocab=None):
    logging.info(
        "Load document from {}; summary from {}.".format(
            doc_filename, sum_filename))

    with open(doc_filename) as docfile:
        docs = docfile.readlines()
        docs = docs[:160]
    with open(sum_filename) as sumfile:
        sums = sumfile.readlines()
        sums = sums[:160]
    #print(len(sums),"len of sums")
    assert len(docs) == len(sums)
    logging.info("Load {num} pairs of data.".format(num=len(docs)))

    docs = list(map(lambda x: x.split(), docs))
    sums = list(map(lambda x: x.split(), sums))

    hidden_label = hidden_label_gen(docs, sums)

    doc_dict = load_dict(doc_dict_path, max_doc_vocab)
    if doc_dict is None:
        doc_dict = create_dict(doc_dict_path, docs, max_doc_vocab)

    sum_dict = load_dict(sum_dict_path, max_sum_vocab)
    if sum_dict is None:
        sum_dict = create_dict(sum_dict_path, sums, max_sum_vocab)

    docid, cover = corpus_map2id(docs, doc_dict[0])
    logging.info(
        "Doc dict covers {:.2f}% words.".format(cover * 100))
    sumid, cover = corpus_map2id(sums, sum_dict[0])
    logging.info(
        "Sum dict covers {:.2f}% words.".format(cover * 100))

    return docid, sumid, doc_dict, sum_dict, hidden_label


def load_valid_data(doc_filename,
                    sum_filename,
                    doc_dict,
                    sum_dict):
    logging.info(
        "Load validation document from {}; summary from {}.".format(
            doc_filename, sum_filename))
    with open(doc_filename) as docfile:
        docs = docfile.readlines()
    with open(sum_filename) as sumfile:
        sums = sumfile.readlines()
    assert len(sums) == len(docs)

    logging.info("Load {} validation documents.".format(len(docs)))

    docs = list(map(lambda x: x.split(), docs))
    sums = list(map(lambda x: x.split(), sums))

    hidden_label = hidden_label_gen(docs, sums)

    docid, cover = corpus_map2id(docs, doc_dict[0])
    logging.info(
        "Doc dict covers {:.2f}% words on validation set.".format(cover * 100))
    sumid, cover = corpus_map2id(sums, sum_dict[0])
    logging.info(
        "Sum dict covers {:.2f}% words on validation set.".format(cover * 100))
    return docid, sumid, hidden_label


def corpus_preprocess(corpus):
    import re
    ret = []
    for line in corpus:
        x = re.sub('\\d', '#', line)
        ret.append(x)
    return ret


def sen_postprocess(sen):
    return sen


def load_test_data(doc_filename, doc_dict):
    logging.info("Load test document from {doc}.".format(doc=doc_filename))

    with open(doc_filename) as docfile:
        docs = docfile.readlines()
    docs = corpus_preprocess(docs)

    logging.info("Load {num} testing documents.".format(num=len(docs)))
    docs = list(map(lambda x: x.split(), docs))

    docid, cover = corpus_map2id(docs, doc_dict[0])
    logging.info(
        "Doc dict covers {:.2f}% words.".format(cover * 100))

    return docs, docid

def hidden_label_gen( diff, commit):
    
    #with open(commit) as docfile:
    #    docs = docfile.readlines()
    #commit= docs.split()
    hidden_label = []
    #^.,()-//'#0-9
    pattern = re.compile("[\w]")
    pattern2 = re.compile("[^0-9]")
    for i in range(len(diff)):
        temp = []
        for j in range(len(diff[i])):
            for k in range(len(commit[i])): commit[i][k] = commit[i][k].lower()
            if diff[i][j].lower() in commit[i] and pattern.match(diff[i][j]) and pattern2.match(diff[i][j]):
                temp.append(1)
            else:
                temp.append(0)
        hidden_label.append(temp)
    return np.asarray(hidden_label)

def add_pad(data, fixlen):
    data = map(lambda x: x + [ID_PAD] * (fixlen - len(x)), data)
    data = list(data)
    return np.asarray(data)

def add_pad_for_hidden(data, fixlen):
    zero_padding = np.zeros((data.shape[0],fixlen), dtype=np.float32)
    for i, seq in enumerate(data):
        length = len(seq)
        zero_padding [i,:length] = np.array(seq)
    return zero_padding


def get_batch(data, bucket, bucket_id, batch_size, ispad, idx):
    encoder_inputs, decoder_inputs=  [], []
    hidden_pad_all =[]
    encoder_len, decoder_len, hid_len = [], [], []

    # Get a random batch of encoder and decoder inputs from data,
    # and add GO to decoder.
    for i in range(batch_size):
        if ispad:
            print("bucket_id",bucket_id," idx",idx,"i",i)
            print("data[bucket_id] shape", np.asarray(data).shape)
            encoder_input, decoder_input, hid_label = data[bucket_id][idx+i]
        else:
            encoder_input, decoder_input, hid_label = random.choice(data[bucket_id])

        encoder_inputs.append(encoder_input)
        encoder_len.append(len(encoder_input))

        decoder_inputs.append(decoder_input)
        decoder_len.append(len(decoder_input))

        hidden_pad_all.append(hid_label)
        hid_len.append(len(hid_label))

        batch_enc_len = max(encoder_len)
        batch_dec_len = max(decoder_len)
        batch_hid_len = bucket[bucket_id][0]


    encoder_inputs = add_pad(encoder_inputs, batch_enc_len)
    decoder_inputs = add_pad(decoder_inputs, batch_dec_len)
    hid_labels = add_pad(hidden_pad_all, batch_hid_len)

    encoder_len = np.asarray(encoder_len)
    hid_labels = np.asarray(hid_labels)
    # decoder_input has both <GO> and <EOS>
    # len(decoder_input)-1 is number of steps in the decoder.
    decoder_len = np.asarray(decoder_len) - 1

    return encoder_inputs, decoder_inputs, encoder_len, decoder_len, hid_labels, hid_len



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')

    docid, sumid, doc_dict, sum_dict, hidd_label = load_data(
        "data/train.article.txt", "data/train.title.txt",
        "data/doc_dict.txt", "data/sum_dict.txt",
        30000, 30000)

    checkid = np.random.randint(len(docid))
    print(checkid)
    print(docid[checkid], sen_map2tok(docid[checkid], doc_dict[1]))
    print(sumid[checkid], sen_map2tok(sumid[checkid], sum_dict[1]))

    docid, sumid, hidid = load_valid_data(
        "data/valid.article.filter.txt", "data/valid.title.filter.txt",
        doc_dict, sum_dict)

    checkid = np.random.randint(len(docid))
    print(checkid)
    print(docid[checkid], sen_map2tok(docid[checkid], doc_dict[1]))
    print(sumid[checkid], sen_map2tok(sumid[checkid], sum_dict[1]))

    docid = load_test_data("data/test.giga.txt", doc_dict)
    checkid = np.random.randint(len(docid))
    print(checkid)
    print(docid[checkid], sen_map2tok(docid[checkid], doc_dict[1]))
