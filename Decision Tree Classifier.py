"""Refrenced Decision Tree classifier code from lecture 4 lesson 7"""
import pandas as pd
from sklearn.model_selection import train_test_split


import  numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.graph_objects as go

import pydotplus
import pydot
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=";") # reading data file and separating data by delimiter specified (;)
data = data.drop(["citric acid"],axis=1)
print(data)
data["quality"] = data["quality"].astype(str) # converting quality column to string from int64 for uses when creating decision tree
for columns in data:
    print(data[columns])
# Create training/testing datasets


train, test = train_test_split(data, test_size=0.2)# splitting data to train and test | data = 80% | testing set = 20%
train_labels = train.iloc[:,-2] # train labels = alcohol content
print(train_labels)


train_data = train.drop(['citric acid','quality'], axis=1)
test_labels = test.iloc[:,-2]
test_data = test.drop(["citric acid","quality"], axis=1)
# For multiclass classification, we will use 'Hobby' as the target
train_labels_mc = train['quality'] # multiclass quality
test_labels_mc = test['quality'] # multiclass = quality


# Train an unconstrained decision tree
from sklearn.tree import DecisionTreeClassifier # imports
X = train_data
y = train_labels_mc
tree_clf = DecisionTreeClassifier(max_depth=5) # creating model of depth 5
tree_clf.fit(X, train_labels_mc) # fitting data to decision tree model of depth 5

# Export tree for GraphViz
from sklearn.tree import export_graphviz
export_graphviz(
tree_clf,
out_file="Decision Tree Classifier.dot",
feature_names=train_data.columns.values,
class_names=tree_clf.classes_,
rounded=True,
filled=True
)
prediction = tree_clf.predict(test_data) # predicting using decision tree
print(accuracy_score(test_labels_mc,prediction)) # printing accuracy for decision tree classifier
print(confusion_matrix(test_labels_mc, prediction)) # printing confusion matrix
print(classification_report(test_labels_mc, prediction)) # printing classification matrix


print(tree_clf.feature_importances_ , "features")  #printing features of importance in relation to quality of wine


