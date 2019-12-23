# -*-coding:utf-8 -*-
from sklearn.model_selection import KFold
import data_processing
from classify import SentimentTrend, vectorizers, classifiers


def KFold_validation(data_X, data_y, vectorizer_method, classify_method):
    """
    Model Comparison
    :param data_X:
    :param data_y:
    :param vectorizer_method:
    :param classify_method:
    :return:
    """
    scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train, test in kf.split(data_X):
        X_train = [data_X[i] for i in train]
        X_test = [data_X[i] for i in test]
        y_train = [data_y[i] for i in train]
        y_test = [data_y[i] for i in test]
        sentiment_trend = SentimentTrend(vectorizer_method=vectorizer_method, classify_method=classify_method)
        sentiment_trend.train(X_train, y_train)
        y_pred = sentiment_trend.test(X_test)
        score = sentiment_trend.evaluate(y_test, y_pred)
        scores.append(score)
    return data_processing.average(scores)


if __name__ == "__main__":
    data_X, data_y = data_processing.load_segment_data()

    # # Model Comparison
    scores_file = open('data/scores.txt', 'w')
    evaluates = ['accuracy', 'pos_precision', 'pos_recall', 'pos_f1_score', 'neg_precision', 'neg_recall', 'neg_f1_score']
    scores_file.write('vectorizer_method, classify_method, ' + ', '.join(evaluates) + '\n')
    for vectorizer_method in vectorizers.keys():
        for classify_method in classifiers.keys():
            score = KFold_validation(data_X, data_y, vectorizer_method=vectorizer_method, classify_method=classify_method)
            print(vectorizer_method, classify_method, score)
            scores_file.write(vectorizer_method + ', ' + classify_method + ', ')
            for item in evaluates:
                scores_file.write(str(score[item]) + ', ')
            scores_file.write('\n')
    scores_file.close()