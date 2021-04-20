#coding=utf-8
import data_loader
import data_processing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = data_loader.DATA_DIR

corpus, labels = data_loader.get_data(DATA_DIR)
stopwords = data_loader.get_stopwords(DATA_DIR)
#train_test_split将数据集按test_size划分测试、训练集， random_state相同每次随机结果都一样
corpus_train, corpus_test, y_train, y_test = train_test_split(corpus,  labels, test_size=0.4,  random_state=1)

norm_train = data_processing.norm_corpus(corpus_train, stopwords)
norm_test = data_processing.norm_corpus(corpus_test, stopwords)
#文本数据转成矩阵形式；CountVectorizer是转换器，保留篇频最小值为2
x_train, x_test, vectorizer = data_processing.convert_data(
                    norm_train, norm_test, CountVectorizer(min_df=2) )

print("全部数据数量：", len(corpus_train) + len(corpus_test))
print("训练数据数量：", len(corpus_train), "\n")
print("分词后的文本样例：\n", norm_train[1])
print("训练集特征词数量：", len(vectorizer.get_feature_names()))


from sklearn.naive_bayes import BernoulliNB
LABELS = ["垃圾邮件","正常邮件"]
def show_prediction(idx_list):
    model = BernoulliNB() #选择模型
    model.fit(x_train, y_train)  #训练模型
    y_pred = model.predict(x_test)  #模型预测，每条记录返回0,1
    for idx in idx_list:
        print("原来的邮件类别：", LABELS[int(y_test[idx])])
        print("预测的邮件类别：", LABELS[int(y_pred[idx])])
        print("正文：\n", corpus_test[idx])
        print("=========================")

show_prediction([0,1])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
def evaluate_models():
    models = { "朴素贝叶斯": BernoulliNB(),"逻辑回归": LogisticRegression()}
    for model_name, model in models.items():
        print("分类器：", model_name)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=LABELS)
        print("混淆矩阵：\n", confusion_matrix(y_test, y_pred))
        print("分类报告\n", report)

evaluate_models()