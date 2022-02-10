from matplotlib.colors import ListedColormap
from sklearn import  datasets, metrics, tree 
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, log_loss, precision_recall_curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from sklearn import  datasets, metrics, tree 
from sklearn.model_selection import train_test_split

bioresponce = pd.read_csv('F:/Documents/ITMO/1курс/Machine_Learning_2022/Lab1/Task_1/bioresponse.csv', header=0, sep=',',encoding='latin1',on_bad_lines='skip')

#prepare data
train_data, test_data,  = train_test_split(bioresponce, test_size = 0.3, random_state = 1)
X_train =  train_data.iloc[:,1:] 
Y_train =  train_data.iloc[:,0]
X_test =    test_data.iloc[:,1:]                                                                       
Y_test =  test_data.iloc[:,0]


#Small Decision tree
clf = tree.DecisionTreeClassifier(random_state=1,max_depth= 10)
clf.fit(X_train, Y_train)

prediction_train_DT_small = clf.predict(X_train)
prediction_test_DT_small = clf.predict(X_test)

prediction_train_DT_small_prob = clf.predict_proba(X_train)
prediction_test_DT_small_prob = clf.predict_proba(X_test)

#precision
precision_score(Y_train, prediction_train_DT_small)
precision_score(Y_test, prediction_test_DT_small)

#recall 
recall_score(Y_train, prediction_train_DT_small)
recall_score(Y_test, prediction_test_DT_small)


#accuracy
accuracy_score(Y_train, prediction_train_DT_small)
accuracy_score(Y_test, prediction_test_DT_small)

#F1-score
f1_score(Y_train, prediction_train_DT_small)
f1_score(Y_test, prediction_test_DT_small)

#Log-loss

log_loss(Y_train, prediction_train_DT_small_prob)
log_loss(Y_test, prediction_test_DT_small_prob)




#Deep Decision tree
clf_deep = tree.DecisionTreeClassifier(random_state=1,max_depth= 50)
clf_deep.fit(X_train, Y_train)

prediction_train_DT_deep = clf_deep.predict(X_train)
prediction_test_DT_deep = clf_deep.predict(X_test)

prediction_train_DT_deep_prob = clf_deep.predict_proba(X_train)
prediction_test_DT_deep_prob = clf_deep.predict_proba(X_test)

#precision
precision_score(Y_train, prediction_train_DT_deep)
precision_score(Y_test, prediction_test_DT_deep)

#recall 
recall_score(Y_train, prediction_train_DT_deep)
recall_score(Y_test, prediction_test_DT_deep)


#accuracy
accuracy_score(Y_train, prediction_train_DT_deep)
accuracy_score(Y_test, prediction_test_DT_deep)

#F1-score
f1_score(Y_train, prediction_train_DT_deep)
f1_score(Y_test, prediction_test_DT_deep)

#Log-loss

log_loss(Y_train, prediction_train_DT_deep_prob)
log_loss(Y_test, prediction_test_DT_deep_prob)



#random forest on small trees
rf_classifier_low_depth = RandomForestClassifier(n_estimators = 50, max_depth = 2, random_state = 1)


rf_classifier_low_depth.fit(X_train, Y_train)

prediction_train_RF_small = rf_classifier_low_depth.predict(X_train)
prediction_test_RF_small = rf_classifier_low_depth.predict(X_test)

prediction_train_RF_small_prob = rf_classifier_low_depth.predict_proba(X_train)[:,1]
prediction_test_RF_small_prob = rf_classifier_low_depth.predict_proba(X_test)[:,1]


#precision
precision_score(Y_train, prediction_train_RF_small)
precision_score(Y_test, prediction_test_RF_small)

#recall 
recall_score(Y_train, prediction_train_RF_small)
recall_score(Y_test, prediction_test_RF_small)

#accuracy
accuracy_score(Y_train, prediction_train_RF_small)
accuracy_score(Y_test, prediction_test_RF_small)

#F1-score
f1_score(Y_train, prediction_train_RF_small)
f1_score(Y_test, prediction_test_RF_small)

#Log-loss
log_loss(Y_train, prediction_train_RF_small_prob)
log_loss(Y_test, prediction_test_RF_small_prob)


#random forest on deep trees
rf_classifier = RandomForestClassifier(n_estimators = 50, max_depth = 10, random_state = 1)

rf_classifier.fit(X_train, Y_train)

prediction_train_RF_deep = rf_classifier.predict(X_train)
prediction_test_RF_deep = rf_classifier.predict(X_test)

prediction_train_RF_deep_prob = rf_classifier.predict_proba(X_train)[:,1]
prediction_test_RF_deep_prob = rf_classifier.predict_proba(X_test)[:,1]

#precision
precision_score(Y_train, prediction_train_RF_deep)
precision_score(Y_test, prediction_test_RF_deep)

#recall 
recall_score(Y_train, prediction_train_RF_deep)
recall_score(Y_test, prediction_test_RF_deep)

#accuracy
accuracy_score(Y_train, prediction_train_RF_deep)
accuracy_score(Y_test, prediction_test_RF_deep)

#F1-score
f1_score(Y_train, prediction_train_RF_deep)
f1_score(Y_test, prediction_test_RF_deep)

#Log-loss
log_loss(Y_train, prediction_train_RF_deep_prob)
log_loss(Y_test, prediction_test_RF_deep_prob)





#precision-recall and ROC curves for your models





precs = []
recs = []
threshs = []
labels = ["DT_small","DT_deep", "RF_small", "RF_deep"]
for actual, predicted in zip([Y_train,Y_train, Y_train, Y_train], 
                                    [prediction_train_DT_small,prediction_train_DT_deep, prediction_train_RF_small, prediction_train_RF_deep]):
    prec, rec, thresh = precision_recall_curve(actual, predicted)
    precs.append(prec)
    recs.append(rec)
    threshs.append(thresh)
plt.figure(figsize=(15, 5))
for i in range(len(labels)):
    ax = plt.subplot(1, len(labels), i+1)
    plt.plot(threshs[i], precs[i][:-1], label="precision")
    plt.plot(threshs[i], recs[i][:-1], label="recall")
    plt.xlabel("threshold")
    ax.set_title(labels[i])
    plt.legend()




precs = []
recs = []
threshs = []
labels = ["DT_small","DT_deep", "RF_small", "RF_deep"]
for actual, predicted in zip([Y_test,Y_test, Y_test, Y_test], 
                                    [prediction_test_DT_small,prediction_test_DT_deep, prediction_test_RF_small, prediction_test_RF_deep]):
    prec, rec, thresh = precision_recall_curve(actual, predicted)
    precs.append(prec)
    recs.append(rec)
    threshs.append(thresh)
plt.figure(figsize=(15, 5))
for i in range(len(labels)):
    ax = plt.subplot(1, len(labels), i+1)
    plt.plot(threshs[i], precs[i][:-1], label="precision")
    plt.plot(threshs[i], recs[i][:-1], label="recall")
    plt.xlabel("threshold")
    ax.set_title(labels[i])
    plt.legend()




from sklearn.metrics import roc_curve, roc_auc_score


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
aucs = ""
for actual, predicted, descr in zip([Y_train,Y_train, Y_train, Y_train], 
                                    [prediction_train_DT_small,prediction_train_DT_deep, prediction_train_RF_small, prediction_train_RF_deep],
                                    ["DT_small","DT_deep", "RF_small", "RF_deep"]):
    fpr, tpr, thr = roc_curve(actual, predicted)
    plt.plot(fpr, tpr, label=descr)
    aucs += descr + ":%3f"%roc_auc_score(actual, predicted) + " "
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc=4)
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.subplot(1, 3, 1)
plt.title("ROC train sample")





plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
aucs = ""
for actual, predicted, descr in zip([Y_test,Y_test, Y_test, Y_test], 
                                    [prediction_test_DT_small,prediction_test_DT_deep, prediction_test_RF_small, prediction_test_RF_deep],
                                    ["DT_small","DT_deep", "RF_small", "RF_deep"]):
    fpr, tpr, thr = roc_curve(actual, predicted)
    plt.plot(fpr, tpr, label=descr)
    aucs += descr + ":%3f"%roc_auc_score(actual, predicted) + " "
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc=4)
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.subplot(1, 3, 1)
plt.title("ROC test sample")






[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]

[{1:1}, {2:5}, {3:1}, {4:1}]



#Avoid False Negative

clf = tree.DecisionTreeClassifier(random_state=1,max_depth= 10,class_weight = {0:.05, 1:.95} )
clf.fit(X_train, Y_train)

prediction_train_DT_small = clf.predict(X_train)
prediction_test_DT_small = clf.predict(X_test)


precision_score(Y_train, prediction_train_DT_small)
recall_score(Y_test, prediction_test_DT_small)


from collections import Counter

dict(Counter(prediction_test_DT_small).items())
