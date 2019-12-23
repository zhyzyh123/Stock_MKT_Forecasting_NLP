# -*-coding:utf-8 -*-
import xlrd
import jieba
import jieba.posseg as pseg
from collections import defaultdict

jieba.load_userdict('data/user_dict')


def load_annotation_data(path):
    sentences = []
    labels = []
    data = xlrd.open_workbook(path).sheets()[0]
    for i in range(1, data.nrows):
        if isinstance(data.cell(i, 2).value, str):
            continue
        if int(data.cell(i, 2).value) == 0:
            continue
        sentences.append(data.cell(i, 0).value.encode('utf-8'))
        labels.append(int(data.cell(i, 2).value))
    return sentences, labels


def prepare_segment_data():
    """

    :return:
    """
    sentences, labels = load_annotation_data('data/annotation_data.xlsx')
    sentences_seg = [segment(sen, False) for sen in sentences]
    neg_file = open('data/annotation_data_after_segment/negative.txt', 'a')
    pos_file = open('data/annotation_data_after_segment/positive.txt', 'a')
    for i in range(len(sentences_seg)):
        if labels[i] == 1:
            pos_file.write(sentences_seg[i] + '\n')
        if labels[i] == -1:
            neg_file.write(sentences_seg[i] + '\n')


def load_segment_data():
    data_X = []
    data_y = []
    with open('data/annotation_data_after_segment/negative.txt', 'r') as f:
        for line in f.readlines():
            data_X.append(line.strip('\n'))
            data_y.append(-1)
    with open('data/annotation_data_after_segment/positive.txt', 'r') as f:
        for line in f.readlines():
            data_X.append(line.strip('\n'))
            data_y.append(1)
    return data_X, data_y


def segment(text, part_of_speech_filter=False):
    """
    Participle (ppl.)，remove punctuations, (partially remove POS)
    :param text:
    :param part_of_speech_filter:
    :return:
    """
    ret = []
    for word, part_of_speech in pseg.cut(text):
        if part_of_speech_filter and not keep_part_of_speech(part_of_speech):
            continue
        if part_of_speech == 'x':
            continue
        ret.append(word)
    return ' '.join(ret)


def get_word_count(sentences_seg, labels):
    """
    Return the words with highly different p/n score
    Example：跌破(fall below) 0.0009 0.0031 0.0129
    :param sentences_seg:
    :param labels:
    :return:
    """
    vocabulary = set()
    word_count = defaultdict(lambda: defaultdict(lambda: 0))
    label_count = defaultdict(lambda: 0)
    for i in range(len(labels)):
        label_count[str(labels[i])] += 1
        for word in set(sentences_seg[i].split(' ')):
            vocabulary.add(word)
            word_count[str(labels[i])][word] += 1
    word_ratio = defaultdict(lambda: {})
    for label, total in label_count.items():
        for word, count in word_count[label].items():
            word_ratio[label][word] = round(count * 1.0 / total, 4)
    for word in vocabulary:
        pos_ratio = word_ratio['1'].get(word, 0)
        neg_ratio = word_ratio['-1'].get(word, 0)
        neu_ratio = word_ratio['0'].get(word, 0)
        if word_count['1'].get(word, 0) <= 5 and word_count['-1'].get(word, 0) <= 5:
            continue
        if (neg_ratio == 0 and pos_ratio >= 0.005) or (pos_ratio == 0 and neg_ratio >= 0.005):
            print(word, pos_ratio, neu_ratio, neg_ratio)
        elif neg_ratio != 0:
            if (pos_ratio / neg_ratio > 4) or (pos_ratio / neg_ratio < 0.25):
                print(word, pos_ratio, neu_ratio, neg_ratio)


def keep_part_of_speech(part_of_speech):
    keep = ['n', 'v', 'a']
    for item in keep:
        if item in part_of_speech:
            return True
    return False


def average(scores):
    score_avg = defaultdict(lambda: 0)
    for item in scores:
        for k, v in item.items():
            score_avg[k] += v
    ret = {}
    for k, v in score_avg.items():
        ret[k] = round(v / len(scores), 4)
    return ret
