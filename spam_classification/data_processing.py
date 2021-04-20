#coding=utf-8
import re
import jieba

def norm_corpus(corpus, stopword_list):
    result_corpus = []
    ##匹配连续2个以上的英文+空格符号， 后面替换成一个空格
    pattern = re.compile('[{}\\s]'.format(re.escape(r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')) + r"{2,}")
    for text in corpus:
        #分词，按停用词表过滤
        seg_text = ' '.join([x.strip() for x in jieba.cut(text) if x.strip() not in stopword_list])
        result_corpus.append(pattern.sub(" ", seg_text))
    return result_corpus


def convert_data(norm_train, norm_test, vectorizer):
    ## fit把数据集中所有文字按规则（默认空格）切分成词元以后每个词元记录一个数字
    ## transform对切分文字匹配出数字id，作为向量维度下标
    ## fit_transform ：两个功能合在一起
    train_features = vectorizer.fit_transform(norm_train)
    test_feature = vectorizer.transform(norm_test)
    return train_features, test_feature, vectorizer