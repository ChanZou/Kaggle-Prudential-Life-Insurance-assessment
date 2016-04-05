import re
import pandas
import numpy
from sklearn import feature_selection, cross_validation
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

__author__ = 'Chan'

insurance = pandas.read_csv("train.csv")

count = 0
mapping = {}
for cate in insurance["Product_Info_2"].unique():
    mapping[cate] = count
    count += 1
P2 = insurance["Product_Info_2"].copy()
for k,v in mapping.iteritems():
    P2[P2==k] = v
insurance["Product_Info_2"] = P2

not_full = []
for name in list(insurance.columns.values):
    if insurance[name].count() != insurance["Id"].count():
        not_full.append(name)

print not_full

medical = re.compile("Medical_History_", re.S)
for name in not_full:
    insurance[name] = insurance[name].fillna(insurance[name].median())
insurance = insurance.drop(["Id"], axis=1)
train = insurance.drop(["Response"], axis=1)
result = insurance["Response"]

selector = feature_selection.SelectKBest(feature_selection.f_classif, k=34)
selector.fit(train, result)

'''
scores = selector.scores_
plt.bar(range(len(scores)), scores)
plt.xticks(range(len(train.columns.values)), list(train.columns.values))
plt.show()
'''

train = train[numpy.array(train.columns)[selector.get_support()]]

randomforest = RandomForestClassifier(random_state=1, n_estimators=300, min_samples_split=20, min_samples_leaf=5,
                                      oob_score=True)

kf = cross_validation.KFold(train.shape[0], n_folds=8)
predictions = []
for cv_train, cv_test in kf:
    train_target = result.iloc[cv_train]
    X = train.iloc[cv_train, :]
    randomforest.fit(X, train_target)
    predictions.append(list(randomforest.predict(train.iloc[cv_test, :])))
    print randomforest.oob_score_

randomforest.fit(train, result)

predictions = numpy.concatenate(predictions, axis=0)
print predictions
score = len(result[result==predictions])/float(len(predictions))
print score

test = pandas.read_csv("test.csv")
ID = test["Id"]
P2 = test["Product_Info_2"].copy()
for k, v in mapping.iteritems():
    P2[P2==k] = v
test["Product_Info_2"] = P2

for name in not_full:
    test[name] = test[name].fillna(test[name].median())

test = test.drop(["Id"], axis=1)
test = test[numpy.array(test.columns)[selector.get_support()]]

predictions = randomforest.predict(test)
submission = pandas.DataFrame({"Id": ID,
                               "Response": predictions})
submission.to_csv("submission.csv", index=False)


