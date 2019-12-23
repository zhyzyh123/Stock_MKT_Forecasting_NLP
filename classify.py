# -*-coding:utf-8 -*-
import os
import xlrd
import math
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import data_processing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


vectorizers = {
    'binary': CountVectorizer(binary=True, min_df=2),
    'count': CountVectorizer(binary=False, min_df=2),
    'tfidf': TfidfVectorizer(min_df=2)
}
classifiers = {
    'LinearSVC': LinearSVC(),
    'LogisticReg': LogisticRegression(solver='lbfgs'),
    'MultinomialNB': MultinomialNB(),
    # # The four models below don't perform very well
    # 'DecisionTree': DecisionTreeClassifier(),
    # 'RandomForest': RandomForestClassifier(),
    # 'AdaBoost': AdaBoostClassifier(base_estimator=LogisticRegression()),
    # 'GBDT': GradientBoostingClassifier(),
}


class SentimentTrend:

    def __init__(self, vectorizer_method='binary', classify_method='MultinomialNB'):
        self.vectorizer = vectorizers[vectorizer_method]
        self.model = classifiers[classify_method]

    def train(self, data_x, data_y):
        features = self.vectorizer.fit_transform(data_x).toarray()
        self.model = self.model.fit(features, data_y)

    def test(self, X_test):
        X_vec = self.vectorizer.transform(X_test).toarray()
        y_pred = self.model.predict(X_vec)
        return y_pred

    def evaluate(self, y_true, y_pred):
        ret = {
                  'accuracy': round(accuracy_score(y_true, y_pred), 4),
                  'pos_precision': round(precision_score(y_true, y_pred, pos_label=1), 4),
                  'pos_recall': round(recall_score(y_true, y_pred, pos_label=1), 4),
                  'pos_f1_score': round(f1_score(y_true, y_pred, pos_label=1), 4),
                  'neg_precision': round(precision_score(y_true, y_pred, pos_label=-1), 4),
                  'neg_recall': round(recall_score(y_true, y_pred, pos_label=-1), 4),
                  'neg_f1_score': round(f1_score(y_true, y_pred, pos_label=-1), 4)
               }
        return ret

    def calculate_BI(self, pos_num, neg_num):
        return round(math.log((1 + pos_num) * 1.0 / (1 + neg_num)), 4)

    def run(self, path):
        date_count = defaultdict(lambda: defaultdict(lambda: 0))
        date_BI = {}
        for _, dir, files in os.walk(path):
            for file in files:
                if (not file.startswith('._')) and file.endswith('xls'):
                    print(file)
                    data = xlrd.open_workbook(os.path.join(_, file)).sheets()[0]
                    for i in range(1, data.nrows):
                        sentence = data.cell(i, 0).value.encode('utf-8')
                        date = data.cell(i, 1).value.split(' ')[0]
                        if len(date) < 2:
                            continue
                        try:
                            year = str(int(data.cell(i, 2).value))
                        except:
                            year = '2019'
                        X_test = self.vectorizer.transform([data_processing.segment(sentence)]).toarray()
                        y_pred = self.model.predict(X_test)[0]
                        date_count[year + '-' + date][str(y_pred)] += 1
        for date, count in date_count.items():
            date_BI[date] = self.calculate_BI(count['1'], count['-1'])
        return date_count, date_BI


def plot_date_BI(date_BI):
    """
    Plot the computed BI Index from 2018-8-07 to 2019-12-17
    """
    months = mdates.MonthLocator()  # every month
    months_fmt = mdates.DateFormatter('%Y-%m')
    date_BI['date'] = pd.to_datetime(date_BI.date)
    fig, ax = plt.subplots()
    ax.plot(date_BI['date'][1:], date_BI['BI'][1:])
    # format the ticks
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(months)
    fig.autofmt_xdate()
    ax.set_xlabel('date')
    ax.set_ylabel('BI')
    fig.set_size_inches(10,5)
    plt.savefig('data/date_BI.png')


sse = pd.read_csv('data/YahooFinance_SSE.csv', usecols = ['Date', 'Close'])
sse.Date = pd.to_datetime(sse.Date)


def plot_compare(window_days):
    """
    Generate four subplots to compare the BI Index and SSE Composite Index
    using different days of rolling window.
    """
    data_compare = date_BI.merge(sse, left_on = 'date', right_on = 'Date', how = 'left').dropna(how='any', axis=0)
    df = data_compare.set_index(data_compare['date'])
    df = df.rolling(window=window_days).mean().dropna(how='any', axis=0)
    # Fixing random state for reproducibility
    t = df.index
    s1 = preprocessing.scale(df.BI)
    s2 = preprocessing.scale(df.Close)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 3)
    ax.plot(t, s1, t, s2)
    ax.set_xlabel('date')
    ax.set_ylabel('BI and SSE Index')
    ax.legend(['BI Index (standardized)', 'SSE Composite Index (standardized)'])
#    cxy, f = ax[1].cohere(s1, s2, 120)
#    ax[1].set_ylabel('coherence')
    fig.tight_layout()
    plt.show()


def plot_compare_smoothing():
    for window_days in [1, 7, 14, 30, 60]:
        print('\nSmoothing by', str(window_days), 'days window')
        plot_compare(window_days)
        

def BI_SSE_correlation():
    data_compare = date_BI.merge(sse, left_on = 'date', right_on = 'Date', how = 'left')
    df = data_compare.dropna(how='any', axis=0)
    df = df.set_index(df['date'])
    max_window = 0
    max_day = 0
    max_corr = 0
    plt.figure(figsize=(8,5))
    for window_days in [1, 7, 14]:
        df1 = df.rolling(window=window_days).mean().dropna(how='any', axis=0)
        days_diff = list(range(1,60))
        corr = []
        for day in days_diff:
            s1 = preprocessing.scale(df1.BI[:-day])
            s2 = preprocessing.scale(df1.Close[day:])
            correlation = pearsonr(s1, s2)[0]
            corr.append(correlation)
            if correlation > max_corr:
                max_window = window_days
                max_day = day
                max_corr = correlation
        plt.plot(days_diff, corr)
    plt.legend([str(x)+' days rolling' for x in [1, 7, 14]])
    plt.scatter(max_day, max_corr)
    plt.annotate("Max Correlatio: %.4f (%.f days)" % (max_corr, max_day),
                 xy=(max_day-10, max_corr+0.09))
    plt.ylim(0,1)
    print('\nmax_window:', max_window, '\nmax_day:', max_day, '\nmax_corr:', round(max_corr,4))


def sse_prediction():
    """
    Based on the BI Index, use the 14-day rolling average to predict the SSE
    Composite Index for the 12 days following 2019-12-17.
    """
    data_compare = date_BI.merge(sse, left_on = 'date', right_on = 'Date', how = 'left')
    df = data_compare.dropna(how='any', axis=0)
    df = df.set_index(df['date'])
    df1 = df.rolling(window=14).mean().dropna(how='any', axis=0)
    df_train = df1[:-12].copy()
    df_train.Close[:] = df1['Close'][12:]
    df_predict = df1[-12:].copy()
    df_predict.Close = np.nan
    X = np.array(df_train['BI']).reshape(-1,1)
    y = df_train['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
    model = KNeighborsRegressor().fit(np.array(X_train).reshape(-1,1), y_train)        
    df_predict['Close'] = model.predict(np.array(df_predict.BI).reshape(-1,1))
    times = pd.date_range('2019-12-18', periods=12, freq='D').to_frame()
    times = times.drop(columns=0)
    times['BI'] = np.nan
    times['Close'] = model.predict(np.array(df_predict.BI).reshape(-1,1))    
    df_predict = pd.concat([df.drop(columns=['date', 'Date']), times])
    df_predict = df_predict.rolling(window=14).mean()[-50:]
    t = df_predict.index
    s1 = preprocessing.scale(df_predict['BI'])
    s2 = preprocessing.scale(df_predict['Close'])
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 3)
    ax.plot(t, s1, t, s2)
    ax.set_xlabel('date')
    ax.set_ylabel('BI and SSE Index')
    ax.legend(['BI Index (standardized)', 'SSE Composite Index (standardized)'])
    plt.axvline(x='2019-12-17', color='orange', linestyle='--')
    plt.annotate("Prediction", xy=('2019-12-21', 1))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # prepare_segment_data() # No need to run again, or will result in duplications
    data_X, data_y = data_processing.load_segment_data()

    # # Go through all the words in the text, compute tf-idf score for positive/
    # # negative attitudes and return the words with with highly different p/n score
    # get_word_count(data_X, data_y)

    # # Choose the best model -- tfidf + LogisticReg, accuracy: 0.84
    # sentiment_trend = SentimentTrend(vectorizer_method='tfidf', classify_method='LogisticReg')
    # sentiment_trend.train(data_X, data_y)
    
    # # Applied all-year data and compute BI Indices. (Sentiment Time Series Data)
    # date_count, date_BI = sentiment_trend.run('data/raw_data')
    # date_BI = sorted(date_BI.items(), key=lambda x: x[0])

    # with open('data/date_BI.csv', 'w') as f:
    #     for item in date_BI:
    #         f.write(item[0] + ',' + str(item[1]) + '\n')

    date_BI = pd.read_csv('data/date_BI.csv', header=None)
    date_BI.columns=['date', 'BI']

    # data
    plot_date_BI(date_BI)
    plot_compare_smoothing()
    plot_compare(14)
    BI_SSE_correlation()
    sse_prediction()