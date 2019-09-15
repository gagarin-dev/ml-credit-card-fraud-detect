
# Fraud Detection

## Task

Build a logistic regression model using Scikit-learn to predict fraudulent transactions by training it on [Credit Card Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) dataset from Kaggle. Before you train the model, create at least 1 visualization of the data.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Scaling by standardisation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
```

## Explore the Dataset / Visualize


```python
df = pd.read_csv('creditcard.csv', low_memory=False)
df = df.sample(frac=1).reset_index(drop=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>127695.0</td>
      <td>-1.180302</td>
      <td>0.997052</td>
      <td>-1.141980</td>
      <td>1.122934</td>
      <td>0.870430</td>
      <td>0.496341</td>
      <td>2.626909</td>
      <td>-0.275357</td>
      <td>-1.251222</td>
      <td>...</td>
      <td>-0.116098</td>
      <td>-0.213810</td>
      <td>-0.495616</td>
      <td>-0.008486</td>
      <td>1.185834</td>
      <td>-0.207036</td>
      <td>0.126568</td>
      <td>-0.016534</td>
      <td>362.00</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>74522.0</td>
      <td>-0.491090</td>
      <td>0.736967</td>
      <td>1.121518</td>
      <td>1.855768</td>
      <td>2.192105</td>
      <td>4.720293</td>
      <td>-0.499471</td>
      <td>1.195404</td>
      <td>-1.064379</td>
      <td>...</td>
      <td>0.093564</td>
      <td>0.267335</td>
      <td>-0.344519</td>
      <td>1.041548</td>
      <td>0.398812</td>
      <td>0.448862</td>
      <td>0.111847</td>
      <td>0.087620</td>
      <td>18.15</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>141751.0</td>
      <td>-1.081088</td>
      <td>-0.891385</td>
      <td>1.263885</td>
      <td>-0.395564</td>
      <td>1.579365</td>
      <td>-1.016444</td>
      <td>-0.654525</td>
      <td>0.249166</td>
      <td>0.485191</td>
      <td>...</td>
      <td>-0.003331</td>
      <td>-0.447941</td>
      <td>0.267298</td>
      <td>0.545437</td>
      <td>-0.103051</td>
      <td>-0.631727</td>
      <td>-0.021950</td>
      <td>0.028339</td>
      <td>1.18</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>131485.0</td>
      <td>2.057686</td>
      <td>-0.038430</td>
      <td>-1.058048</td>
      <td>0.417849</td>
      <td>-0.127807</td>
      <td>-1.212282</td>
      <td>0.204981</td>
      <td>-0.350565</td>
      <td>0.502264</td>
      <td>...</td>
      <td>-0.284453</td>
      <td>-0.677820</td>
      <td>0.336365</td>
      <td>0.053365</td>
      <td>-0.291787</td>
      <td>0.194302</td>
      <td>-0.069472</td>
      <td>-0.058925</td>
      <td>3.96</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>38305.0</td>
      <td>0.658722</td>
      <td>-2.234003</td>
      <td>1.721964</td>
      <td>0.101310</td>
      <td>-2.625252</td>
      <td>0.696838</td>
      <td>-1.609055</td>
      <td>0.428602</td>
      <td>0.870870</td>
      <td>...</td>
      <td>0.496141</td>
      <td>1.149756</td>
      <td>-0.402166</td>
      <td>0.628406</td>
      <td>0.346473</td>
      <td>0.039347</td>
      <td>0.042984</td>
      <td>0.071146</td>
      <td>298.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



#### Check if Any Missing Values in the Dataset


```python
df.isnull().sum()
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64



No missing values were detected.


```python
fraud = df.loc[df['Class'] == 1]
non_fraud = df.loc[df['Class'] == 0]
print(f'Number of non-fraudulent operations: {len(non_fraud)}')
print(f'Number of fraudulent operations: {len(fraud)}')
print(f"The average amount of fradulent transactions is {fraud['Amount'].mean()}")
print(f"The maximum amount of fradulent transactions is {fraud['Amount'].max()}") 
```

    Number of non-fraudulent operations: 284315
    Number of fraudulent operations: 492
    The average amount of fradulent transactions is 122.21132113821143
    The maximum amount of fradulent transactions is 2125.87


## Visualize the Dataset


```python
# Plot the Amount vs Class distribution
plt.scatter(df['Amount'],df['Class'])
plt.title('Amount vs Class distribution')
plt.xlabel('Amount (currency)')
plt.ylabel('Class: (1)-fraud, (0)-legitimate')
plt.show()
```


![png](output_9_0.png)


## Decide Which Features Are Important

In this analysis we will not use the "Time" feature. Our matrix "X" of independent variables will consist of the features "V1" through "V28" and also include the "Amount" feature. The dependent variable vector "y" will consist the values of the "Class" column of the dataset.


```python
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
```

## Spliting the dataset into the Training set (85%) and Test set (15%)



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
```

## Feature Scaling


```python
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Need to fit the object into our feature set, then transform feature set
X_test = sc_X.transform(X_test) # Only need to transform, because the object is already fitted (prev. line), otherwise different scales.
```

## Perform Machine Learning


```python
# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)  # using the same random_state
classifier.fit(X_train, y_train)
```

## Test the Model On the Testing Set


```python
# Predicting the Test set results
y_pred = classifier.predict(X_test)
```

#### Evaluate Performance of the Model


```python
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

    [[42642     5]
     [   25    50]]



```python
print(f'Number of correct predictions: {cm[0,0] + cm[1,1]}')
print(f'Number of incorrect predictions: {cm[0,1] + cm[1,0]}')
```

    Number of correct predictions: 42692
    Number of incorrect predictions: 30


Very good!
The performance score of the model on the Test Set:


```python
print(classifier.score(X_test,y_test))
```

    0.9992977856841908


But if we remember the main purpose of this model - to detect fradulent credit card transactions, we should rather be interested in how well the model can detect fradulent transactions specifically. For that we will use the __recall score__ - the ability of the classifier to find all the positive (fradulent) samples: 


```python
print(recall_score(y_test, y_pred))
```

    0.6666666666666666


This is not a bad number, but in order for the model to be used in production the __recall score__ must be significally improved.

Finally the full Classification Report with other performance scores of the model


```python
class_report = classification_report(y_test, y_pred)
print(class_report)
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     42647
               1       0.91      0.67      0.77        75
    
        accuracy                           1.00     42722
       macro avg       0.95      0.83      0.88     42722
    weighted avg       1.00      1.00      1.00     42722
    

