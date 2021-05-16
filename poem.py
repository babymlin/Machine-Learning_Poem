# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:24:19 2021

@author: babymlin
"""

import jieba
import pandas as pd

X_train = pd.read_csv("poem_train.csv")
X_test = pd.read_csv("poem_test.csv")
y_train = X_train["作者"]
y_test = X_test["作者"]

# temp = X_train.iloc[1, 2]
# strings = " ".join(list(jieba.cut(temp)))
# print(X_train["內容"])

def poem2split(s):
    s = s.replace("\r", "").replace("\n", "")
    return " ".join(jieba.cut(s))
# print(poem2split(temp))
X_train["內容"] = X_train["內容"].apply(poem2split)
X_test["內容"] = X_test["內容"].apply(poem2split)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# trans = {"李白":1, "杜甫":2, "白居易":3}
# y_train = y_train.replace(trans)
# y_test = y_test.replace(trans)

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
tranvec = vec.fit_transform(X_train["內容"])
testvec = vec.transform(X_test["內容"])
# print(tranvec)
# temp = vec.vocabulary_

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(tranvec, y_train)

predict = clf.predict(testvec)
print("預測結果:", list(predict))
print("正確結果:", list(y_test))
print("正確率:", clf.score(testvec, y_test))

def poem_predict():
    inp = input("請輸入一首詩:")
    inpsplit = poem2split(inp)
    inpvec = vec.transform([inpsplit])
    return le.inverse_transform(clf.predict(inpvec))[0]

if __name__ == "__main__":
    print(poem_predict())
    




