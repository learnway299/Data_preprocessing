
import numpy as np

DATA_DIR = "./data"

def get_data(path = DATA_DIR):
    '''
    获取数据
    :return: 文本数据，对应的labels
    '''
    with open(path + "./ham_data.txt", encoding="utf8") as ham_f, \
            open(path + "./spam_data.txt", encoding="utf8") as spam_f:
        ham_data = [x.strip() for x in ham_f.readlines() if x.strip()]
        spam_data = [x.strip() for x in spam_f.readlines() if x.strip()]

        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()

        corpus = ham_data + spam_data
        labels = ham_label + spam_label

    return corpus, labels

def get_stopwords(path = DATA_DIR):
    with open(path + "./stop_words.utf8", encoding="utf8") as f:
        stopword_list = set([x.strip() for x in f.readlines()] +
                            list(r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    return  stopword_list
