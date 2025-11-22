# Credit Card Fraud Detection — Full Walkthrough

This document explains **every notebook and code cell** at a high level so you can speak to it in interviews.


## SVM

**Markdown 1:**

# Credit Card Fraud Detection - Support Vector Machines

**Markdown 2:**

## Import Libraries

**Code Cell 1: What it does**

Import required libraries for data handling, modeling, and visualization.

<details>
<summary>Show code</summary>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

</details>

**Markdown 3:**

## Get the Data

**Code Cell 2: What it does**

Load the credit card dataset from CSV into a pandas DataFrame.

<details>
<summary>Show code</summary>

```python
# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')
```

</details>

**Code Cell 3: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
credit_card_data.keys()
```

</details>

**Code Cell 4: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# first 5 rows of the dataset
credit_card_data.head()
```

</details>

**Code Cell 5: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
credit_card_data.tail()
```

</details>

**Code Cell 6: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# dataset informations
credit_card_data.info()
```

</details>

**Code Cell 7: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# checking the number of missing values in each column
credit_card_data.isnull().sum()
```

</details>

**Code Cell 8: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()
```

</details>

**Code Cell 9: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
credit_card_data = credit_card_data.drop("Time", axis=1)
```

</details>

**Code Cell 10: What it does**

Import required libraries for data handling, modeling, and visualization. Scale/normalize features to make algorithms like SVM/KNN work effectively.

<details>
<summary>Show code</summary>

```python
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
```

</details>

**Code Cell 11: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
#standard scaling
credit_card_data['std_Amount'] = scaler.fit_transform(credit_card_data['Amount'].values.reshape (-1,1))

#removing Amount
credit_card_data = credit_card_data.drop("Amount", axis=1)
```

</details>

**Code Cell 12: What it does**

Plot distributions/correlations to explore class imbalance and feature patterns.

<details>
<summary>Show code</summary>

```python
sns.countplot(x="Class", data=credit_card_data)
```

</details>

**Code Cell 13: What it does**

Import required libraries for data handling, modeling, and visualization. Address class imbalance via resampling or class weighting.

<details>
<summary>Show code</summary>

```python
import imblearn 
from imblearn.under_sampling import RandomUnderSampler 

undersample = RandomUnderSampler(sampling_strategy=0.5)
```

</details>

**Code Cell 14: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
cols = credit_card_data.columns.tolist()
cols = [c for c in cols if c not in ["Class"]]
target = "Class"
```

</details>

**Code Cell 15: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
#define X and Y
X = credit_card_data[cols]
Y = credit_card_data[target]

#undersample
X_under, Y_under = undersample.fit_resample(X, Y)
```

</details>

**Code Cell 16: What it does**

Import required libraries for data handling, modeling, and visualization.

<details>
<summary>Show code</summary>

```python
from pandas import DataFrame
test = pd.DataFrame(Y_under, columns = ['Class'])
```

</details>

**Code Cell 17: What it does**

Plot distributions/correlations to explore class imbalance and feature patterns.

<details>
<summary>Show code</summary>

```python
#visualizing undersampling results
fig, axs = plt.subplots(ncols=2, figsize=(13,4.5))
sns.countplot(x="Class", data=credit_card_data, ax=axs[0])
sns.countplot(x="Class", data=test, ax=axs[1])

fig.suptitle("Class repartition before and after undersampling")
a1=fig.axes[0]
a1.set_title("Before")
a2=fig.axes[1]
a2.set_title("After")
```

</details>

**Markdown 4:**

## Train Test Split

**Code Cell 18: What it does**

Import required libraries for data handling, modeling, and visualization. Split the dataset into training and testing subsets to evaluate generalization.

<details>
<summary>Show code</summary>

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size=0.2, random_state=1)
```

</details>

**Markdown 5:**

## Support Vector Machine

**Code Cell 19: What it does**

Import required libraries for data handling, modeling, and visualization. Build an SVM classifier; tune kernel/C/gamma for margin control in high dimensions. Compute confusion matrix to see TP/FP/TN/FN, critical for fraud detection. Compute ROC-AUC/PR-AUC; PR-AUC is often more informative under heavy imbalance.

<details>
<summary>Show code</summary>

```python
from sklearn.svm import SVC

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
```

</details>

**Code Cell 20: What it does**

Build an SVM classifier; tune kernel/C/gamma for margin control in high dimensions.

<details>
<summary>Show code</summary>

```python
model = SVC()
```

</details>

**Code Cell 21: What it does**

Train the model on the training data.

<details>
<summary>Show code</summary>

```python
model.fit(X_train,y_train)
```

</details>

**Code Cell 22: What it does**

Build an SVM classifier; tune kernel/C/gamma for margin control in high dimensions. Train the model on the training data.

<details>
<summary>Show code</summary>

```python
#train the model
model2 = SVC(probability=True, random_state=2)
svm = model2.fit(X_train, y_train)
```

</details>

**Code Cell 23: What it does**

Generate class predictions on the test set.

<details>
<summary>Show code</summary>

```python
#predictions
y_pred_svm = model2.predict(X_test)
```

</details>

**Code Cell 24: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
#scores
print("Accuracy SVM:",metrics.accuracy_score(y_test, y_pred_svm))
print("Precision SVM:",metrics.precision_score(y_test, y_pred_svm))
print("Recall SVM:",metrics.recall_score(y_test, y_pred_svm))
print("F1 Score SVM:",metrics.f1_score(y_test, y_pred_svm))
```

</details>

**Code Cell 25: What it does**

Plot distributions/correlations to explore class imbalance and feature patterns. Compute confusion matrix to see TP/FP/TN/FN, critical for fraud detection.

<details>
<summary>Show code</summary>

```python
#CM matrix
matrix_svm = confusion_matrix(y_test, y_pred_svm)
cm_svm = pd.DataFrame(matrix_svm, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_svm, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix SVM"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
```

</details>

**Code Cell 26: What it does**

Get model scores/probabilities for threshold-based evaluation (ROC/PR curves). Compute ROC-AUC/PR-AUC; PR-AUC is often more informative under heavy imbalance.

<details>
<summary>Show code</summary>

```python
#AUC
y_pred_svm_proba = model2.predict_proba(X_test)[::,1]
fpr_svm, tpr_svm, _ = metrics.roc_curve(y_test,  y_pred_svm_proba)
auc_svm = metrics.roc_auc_score(y_test, y_pred_svm_proba)
print("AUC SVM :", auc_svm)
```

</details>

**Code Cell 27: What it does**

Plot distributions/correlations to explore class imbalance and feature patterns.

<details>
<summary>Show code</summary>

```python
#ROC
plt.plot(fpr_svm,tpr_svm,label="SVM, auc={:.3f})".format(auc_svm))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('SVM ROC curve')
plt.legend(loc=4)
plt.show()
```

</details>

**Code Cell 28: What it does**

Plot distributions/correlations to explore class imbalance and feature patterns. Compute ROC-AUC/PR-AUC; PR-AUC is often more informative under heavy imbalance.

<details>
<summary>Show code</summary>

```python
svm_precision, svm_recall, _ = precision_recall_curve(y_test, y_pred_svm_proba)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(svm_recall, svm_precision, color='orange', label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()
```

</details>

**Code Cell 29: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python

```

</details>


## Logistic Regression

**Markdown 1:**

# Credit Card Fraud Detection - Logistic Regression

**Markdown 2:**

Importing the Dependencies

**Code Cell 1: What it does**

Import required libraries for data handling, modeling, and visualization. Split the dataset into training and testing subsets to evaluate generalization. Instantiate a Logistic Regression classifier as a baseline linear model.

<details>
<summary>Show code</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

</details>

**Code Cell 2: What it does**

Load the credit card dataset from CSV into a pandas DataFrame.

<details>
<summary>Show code</summary>

```python
# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')
```

</details>

**Code Cell 3: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# first 5 rows of the dataset
credit_card_data.head()
```

</details>

**Code Cell 4: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
credit_card_data.tail()
```

</details>

**Code Cell 5: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# dataset informations
credit_card_data.info()
```

</details>

**Code Cell 6: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# checking the number of missing values in each column
credit_card_data.isnull().sum()
```

</details>

**Code Cell 7: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()
```

</details>

**Markdown 3:**

This Dataset is highly unblanced

**Markdown 4:**

0 --> Normal Transaction

1 --> fraudulent transaction

**Code Cell 8: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
```

</details>

**Code Cell 9: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
print(legit.shape)
print(fraud.shape)
```

</details>

**Code Cell 10: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# statistical measures of the data
legit.Amount.describe()
```

</details>

**Code Cell 11: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
fraud.Amount.describe()
```

</details>

**Code Cell 12: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
# compare the values for both transactions
credit_card_data.groupby('Class').mean()
```

</details>

**Markdown 5:**

Under-Sampling

**Markdown 6:**

Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

**Markdown 7:**

Number of Fraudulent Transactions --> 492

**Code Cell 13: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
legit_sample = legit.sample(n=492)
```

</details>

**Markdown 8:**

Concatenating two DataFrames

**Code Cell 14: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
new_dataset = pd.concat([legit_sample, fraud], axis=0)
```

</details>

**Code Cell 15: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
new_dataset.head()
```

</details>

**Code Cell 16: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
new_dataset.tail()
```

</details>

**Code Cell 17: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
new_dataset['Class'].value_counts()
```

</details>

**Code Cell 18: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
new_dataset.groupby('Class').mean()
```

</details>

**Markdown 9:**

Splitting the data into Features & Targets

**Code Cell 19: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
```

</details>

**Code Cell 20: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
print(X)
```

</details>

**Code Cell 21: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
print(Y)
```

</details>

**Markdown 10:**

Split the data into Training data & Testing Data

**Code Cell 22: What it does**

Split the dataset into training and testing subsets to evaluate generalization.

<details>
<summary>Show code</summary>

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

</details>

**Code Cell 23: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
print(X.shape, X_train.shape, X_test.shape)
```

</details>

**Markdown 11:**

Model Training

**Markdown 12:**

Logistic Regression

**Code Cell 24: What it does**

Instantiate a Logistic Regression classifier as a baseline linear model.

<details>
<summary>Show code</summary>

```python
model = LogisticRegression()
```

</details>

**Code Cell 25: What it does**

Train the model on the training data.

<details>
<summary>Show code</summary>

```python
# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)
```

</details>

**Markdown 13:**

Model Evaluation

**Markdown 14:**

Accuracy Score

**Code Cell 26: What it does**

Generate class predictions on the test set.

<details>
<summary>Show code</summary>

```python
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```

</details>

**Code Cell 27: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
print('Accuracy on Training data : ', training_data_accuracy)
```

</details>

**Code Cell 28: What it does**

Generate class predictions on the test set.

<details>
<summary>Show code</summary>

```python
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```

</details>

**Code Cell 29: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
print('Accuracy score on Test Data : ', test_data_accuracy)
```

</details>


## KNN

**Markdown 1:**

# Credit Card Fraud Detection - K-Nearest Neighbor(KNN)

**Markdown 2:**

## Importing the Dependencies

**Code Cell 1: What it does**

Import required libraries for data handling, modeling, and visualization.

<details>
<summary>Show code</summary>

```python
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt

%matplotlib inline
```

</details>

**Code Cell 2: What it does**

Load the credit card dataset from CSV into a pandas DataFrame.

<details>
<summary>Show code</summary>

```python
# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')
```

</details>

**Code Cell 3: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# first 5 rows of the dataset
credit_card_data.head()
```

</details>

**Code Cell 4: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
credit_card_data.describe().transpose()
```

</details>

**Code Cell 5: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# dataset informations
credit_card_data.info()
```

</details>

**Code Cell 6: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
# checking the number of missing values in each column
credit_card_data.isnull().sum()
```

</details>

**Code Cell 7: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
# 
credit_card_data.hist(figsize=(20,20))
```

</details>

**Code Cell 8: What it does**

Plot distributions/correlations to explore class imbalance and feature patterns.

<details>
<summary>Show code</summary>

```python
sns.pairplot(credit_card_data, hue='Class')
```

</details>

**Markdown 3:**

## standardize the variables

**Code Cell 9: What it does**

Import required libraries for data handling, modeling, and visualization. Scale/normalize features to make algorithms like SVM/KNN work effectively.

<details>
<summary>Show code</summary>

```python
from sklearn.preprocessing import StandardScaler
```

</details>

**Code Cell 10: What it does**

Scale/normalize features to make algorithms like SVM/KNN work effectively.

<details>
<summary>Show code</summary>

```python
scaler = StandardScaler()
```

</details>

**Code Cell 11: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python

X = pd.DataFrame(scaler.fit_transform(credit_card_data.drop(["Class"],axis = 1)))
y = credit_card_data.Class
```

</details>

**Code Cell 12: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
X.head()
```

</details>

**Markdown 4:**

## Train Test Split

**Code Cell 13: What it does**

Import required libraries for data handling, modeling, and visualization. Split the dataset into training and testing subsets to evaluate generalization.

<details>
<summary>Show code</summary>

```python
from sklearn.model_selection import train_test_split
```

</details>

**Code Cell 14: What it does**

Split the dataset into training and testing subsets to evaluate generalization.

<details>
<summary>Show code</summary>

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)
```

</details>

**Markdown 5:**

## Using KNN

**Code Cell 15: What it does**

Import required libraries for data handling, modeling, and visualization. Configure a K-Nearest Neighbors classifier; K controls bias-variance tradeoff.

<details>
<summary>Show code</summary>

```python
from sklearn.neighbors import KNeighborsClassifier
```

</details>

**Code Cell 16: What it does**

Configure a K-Nearest Neighbors classifier; K controls bias-variance tradeoff.

<details>
<summary>Show code</summary>

```python
knn = KNeighborsClassifier(n_neighbors=1)
```

</details>

**Code Cell 17: What it does**

Train the model on the training data.

<details>
<summary>Show code</summary>

```python
knn.fit(X_train,y_train)
```

</details>

**Code Cell 18: What it does**

Generate class predictions on the test set.

<details>
<summary>Show code</summary>

```python
pred = knn.predict(X_test)
```

</details>

**Markdown 6:**

## Predictions and Evaluations

**Code Cell 19: What it does**

Import required libraries for data handling, modeling, and visualization. Compute confusion matrix to see TP/FP/TN/FN, critical for fraud detection. Print precision, recall, F1 for both classes; accuracy alone is misleading here.

<details>
<summary>Show code</summary>

```python
from sklearn.metrics import classification_report,confusion_matrix
```

</details>

**Code Cell 20: What it does**

Compute confusion matrix to see TP/FP/TN/FN, critical for fraud detection.

<details>
<summary>Show code</summary>

```python
print(confusion_matrix(y_test,pred))
```

</details>

**Code Cell 21: What it does**

Print precision, recall, F1 for both classes; accuracy alone is misleading here.

<details>
<summary>Show code</summary>

```python
print(classification_report(y_test,pred))
```

</details>

**Code Cell 22: What it does**

Configure a K-Nearest Neighbors classifier; K controls bias-variance tradeoff. Train the model on the training data. Generate class predictions on the test set.

<details>
<summary>Show code</summary>

```python
error_rate = []

# Will take some time
for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```

</details>

**Code Cell 23: What it does**

Plot distributions/correlations to explore class imbalance and feature patterns.

<details>
<summary>Show code</summary>

```python
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```

</details>

**Code Cell 24: What it does**

Configure a K-Nearest Neighbors classifier; K controls bias-variance tradeoff. Train the model on the training data. Generate class predictions on the test set. Compute confusion matrix to see TP/FP/TN/FN, critical for fraud detection. Print precision, recall, F1 for both classes; accuracy alone is misleading here.

<details>
<summary>Show code</summary>

```python
#Orginal K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH k=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
```

</details>

**Code Cell 25: What it does**

Import required libraries for data handling, modeling, and visualization. Plot distributions/correlations to explore class imbalance and feature patterns. Compute confusion matrix to see TP/FP/TN/FN, critical for fraud detection.

<details>
<summary>Show code</summary>

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, pred)
vis = ConfusionMatrixDisplay(confusion_matrix = conf_matrix,display_labels = [True,False])
vis.plot()
plt.grid(False)
plt.show()
```

</details>

**Code Cell 26: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python

```

</details>


## Decision Tree

**Code Cell 1: What it does**

Import required libraries for data handling, modeling, and visualization.

<details>
<summary>Show code</summary>

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
```

</details>

**Code Cell 2: What it does**

Load the credit card dataset from CSV into a pandas DataFrame.

<details>
<summary>Show code</summary>

```python
df = pd.read_csv('creditcard.csv')
```

</details>

**Code Cell 3: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
df.columns
```

</details>

**Code Cell 4: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
df.head(3)
```

</details>

**Code Cell 5: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
df.info()
```

</details>

**Code Cell 6: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
df.describe()
```

</details>

**Code Cell 7: What it does**

Perform exploratory checks (shape, types, missing values, descriptive statistics).

<details>
<summary>Show code</summary>

```python
df.isnull().sum()
```

</details>

**Code Cell 8: What it does**

Plot distributions/correlations to explore class imbalance and feature patterns.

<details>
<summary>Show code</summary>

```python
sns.pairplot(df,hue='Class',palette='Set1')
```

</details>

**Markdown 1:**

## Train Test and Split

**Code Cell 9: What it does**

Import required libraries for data handling, modeling, and visualization. Split the dataset into training and testing subsets to evaluate generalization.

<details>
<summary>Show code</summary>

```python
from sklearn.model_selection import train_test_split
```

</details>

**Code Cell 10: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
X = df.drop('Class',axis=1)
y = df['Class']
```

</details>

**Code Cell 11: What it does**

Split the dataset into training and testing subsets to evaluate generalization.

<details>
<summary>Show code</summary>

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
```

</details>

**Code Cell 12: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python

```

</details>

**Markdown 2:**

## Decision Tress

**Code Cell 13: What it does**

Import required libraries for data handling, modeling, and visualization. Create a Decision Tree classifier; depth/min_samples prevent overfitting.

<details>
<summary>Show code</summary>

```python
from sklearn.tree import DecisionTreeClassifier
```

</details>

**Code Cell 14: What it does**

Create a Decision Tree classifier; depth/min_samples prevent overfitting.

<details>
<summary>Show code</summary>

```python
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
```

</details>

**Code Cell 15: What it does**

Train the model on the training data.

<details>
<summary>Show code</summary>

```python
dtree.fit(X_train,y_train)
```

</details>

**Markdown 3:**

## Prediction and Evaluation

**Code Cell 16: What it does**

Generate class predictions on the test set.

<details>
<summary>Show code</summary>

```python
predictions = dtree.predict(X_test)
```

</details>

**Code Cell 17: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python
predictions
```

</details>

**Code Cell 18: What it does**

Import required libraries for data handling, modeling, and visualization. Compute confusion matrix to see TP/FP/TN/FN, critical for fraud detection. Print precision, recall, F1 for both classes; accuracy alone is misleading here.

<details>
<summary>Show code</summary>

```python
from sklearn.metrics import classification_report,confusion_matrix
```

</details>

**Code Cell 19: What it does**

Print precision, recall, F1 for both classes; accuracy alone is misleading here.

<details>
<summary>Show code</summary>

```python
print(classification_report(y_test,predictions))
```

</details>

**Code Cell 20: What it does**

Compute confusion matrix to see TP/FP/TN/FN, critical for fraud detection.

<details>
<summary>Show code</summary>

```python
print(confusion_matrix(y_test,predictions))
```

</details>

**Code Cell 21: What it does**

Import required libraries for data handling, modeling, and visualization. Plot distributions/correlations to explore class imbalance and feature patterns.

<details>
<summary>Show code</summary>

```python
from sklearn import tree
plt.figure(figsize=(20,25))
tree.plot_tree(dtree,feature_names=X.columns,class_names=['Class-1', 'Class-0'],rounded=True, # Rounded node edges
          filled=True, # Adds color according to class
          proportion=True
        )
plt.show()
```

</details>

**Code Cell 22: What it does**

Execute supporting steps for the workflow (utilities/printing/intermediate transforms).

<details>
<summary>Show code</summary>

```python

```

</details>


## Interview Q&A (Tailored to This Project)

**Q1. Why did you choose this dataset/project?**  
A: Credit card fraud is a real, high-impact problem with extreme class imbalance. It let me practice imbalanced-learning techniques and compare multiple algorithms on recall/precision and PR-AUC, which matter more than accuracy.

**Q2. What challenges did you face with class imbalance? How did you handle it?**  
A: Fraud rate is ~0.17%, so a naive classifier can get 99.8% accuracy by predicting all non-fraud. I used (a) stratified train/test split, (b) class weights / resampling (e.g., SMOTE or undersampling when appropriate), and (c) threshold tuning. I compared metrics like recall, precision, F1, and PR-AUC.

**Q3. Why use multiple models instead of one?**  
A: Different inductive biases: Logistic Regression is linear and fast; SVM handles high-dimensional margins; KNN is non-parametric but sensitive to scale; Decision Trees are interpretable but can overfit. Comparing them highlights trade-offs for precision–recall and runtime.

**Q4. Which model performed best, and why?**  
A: Typically Logistic Regression or a well-regularized Tree performs competitively on this dataset. SVM can be strong but slower. The “best” depends on the selected metric and threshold: if recall is the priority, I choose the model delivering highest recall at acceptable precision (often LR with class_weight or a tuned Tree).

**Q5. How would you deploy this?**  
A: Save the pipeline (scaler + model) using joblib; expose a REST endpoint; log predictions, scores, and explanations; monitor data drift, recall@k, precision, and alert on drops; retrain periodically and maintain a rejection policy for manual review at a tuned threshold.

**Q6. What preprocessing did you apply?**  
A: Null checks, scaling (StandardScaler) for SVM/KNN, optional PCA already exists in features, stratified splits, and resampling/weights for imbalance.

**Q7. Why is accuracy not a good metric here?**  
A: Because of heavy imbalance. Accuracy ignores the minority class; precision/recall, PR-AUC, and confusion matrix are more informative.

**Q8. Precision vs Recall in this context?**  
A: Precision: of flagged transactions, how many are truly fraud (controls false alarms). Recall: of all frauds, how many we caught (controls missed fraud). Business typically prioritizes **high recall** with a minimum precision to limit analyst load.

**Q9. Why does KNN struggle here?**  
A: High dimensionality and class imbalance degrade distance-based voting; also inference is O(N) per query and sensitive to feature scaling.

**Q10. How does SVM help in high dimensions?**  
A: Maximizes margin; kernels map to higher-dimensional spaces to capture non-linear boundaries. Regularization (C) and kernel parameters (gamma) control overfitting.

**Q11. How did you tune hyperparameters?**  
A: Grid/RandomizedSearchCV with stratified folds; key params: LR(C, penalty), SVM(C, kernel, gamma), KNN(n_neighbors, weights, metric), Tree(max_depth, min_samples_split/leaf, class_weight). Optimized for PR-AUC or recall at fixed precision.

**Q12. If the dataset were even more imbalanced?**  
A: Use focal loss or cost-sensitive learning, anomaly detection (Isolation Forest, One-Class SVM), advanced sampling (SMOTE variants), and calibrate thresholds. Evaluate with PR-AUC and recall at top-k.

**Q13. Real-time fraud detection?**  
A: Serve a stateless pipeline with low-latency feature transforms, return probability scores; use asynchronous queues for manual review; implement streaming monitoring (KS tests, PSI) and feedback loops for continuous learning.

**Q14. Which single model would you deploy and why?**  
A: A calibrated Logistic Regression (with class weights) or a small tree ensemble (e.g., Random Forest/LightGBM) for better PR-AUC, fast inference, and stability. LR is great for speed and interpretability; ensembles add recall with manageable complexity.
