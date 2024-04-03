# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection ,import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SANTHOSH T
Register Number:  212223220100
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

### Data.head():
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/148958618/edb65635-f571-45a9-8484-131fe655e915)



### Data.info():
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/148958618/61487aaf-9c35-4bd4-935b-d33fa7ca28a4)



### isnull() and sum():
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/148958618/60224a59-fd2f-4856-832c-83946797cd0e)


### Data value coounts():

![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/148958618/ff383ab9-91c2-4854-8346-60f640f15f2f)


### Data.head() for salary:
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/148958618/007f569a-7ab5-43bb-a2f1-25ae536ec6c2)



### X.head():
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/148958618/27897b4f-6f2e-49b9-898f-57639a10e6ba)


### Accuracy value:
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/148958618/daef7422-1e88-410b-93a4-dd41d0d5dfbc)


### Data prediction:
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/148958618/954b3814-f3cf-4d42-a8cb-0539ef7d115f)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
