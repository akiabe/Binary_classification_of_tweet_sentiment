import pandas as pd

import lightgbm as lgb

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfidf_vec = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
    )
    tfidf_vec.fit(df_train.text.values)

    xtrain = tfidf_vec.transform(df_train.text.values)
    xvalid = tfidf_vec.transform(df_valid.text.values)

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    clf = lgb.LGBMClassifier()
    clf.fit(xtrain, ytrain)
    pred = clf.predict_proba(xvalid)[:, 1]

    auc = metrics.roc_auc_score(yvalid, pred)
    print(f"fold={fold}, auc={auc}")

if __name__ == "__main__":
    for j in range(5):
        run(j)


