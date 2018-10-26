from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def get_onehot_data(ngram):
    ngram_range = {
        3:(3,3),
        2:(2,2),
        1:(1,1)}[ngram]
    df_tr = pd.read_csv('dataset/train.csv', encoding='utf8')
    df_test = pd.read_csv('dataset/test.csv', encoding='utf8')

    df_tr.fillna('pad', inplace=True)
    df_test.fillna('pad', inplace=True)

    param = {
        'ngram_range' : ngram_range,
        'decode_error' : 'ignore',
        'token_pattern' : r'\b\w+\b',
        'analyzer' : 'char',
    }
    vectorizer = CountVectorizer(**param)
    x_tr = vectorizer.fit_transform(df_tr['x'].values).toarray()
    x_test = vectorizer.transform(df_test['x'].values).toarray()

    x_tr[x_tr != 0] = 1
    x_test[x_test != 0] = 1

    y_tr = df_tr['label'].values
    y_test = df_test['label'].values


    return x_tr, x_test, y_tr, y_test