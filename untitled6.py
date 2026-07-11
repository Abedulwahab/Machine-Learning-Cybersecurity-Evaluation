
def Accuracy(y_test,y_pred):
    from sklearn.metrics import confusion_matrix as CM
    cm=CM(y_test,y_pred)
    degree=cm.sum()
    
    S=0
    for i in range(len(cm)):
        for j in range(len(cm)):
            if(i==j):
                S=S+cm[i][j]
    
    print("Accuracy: ",int((S/degree)*100),"%")







import pandas as pd
import numpy as np 

data1=pd.read_csv("Dataset1Train.csv")
data2=pd.read_csv("Dataset1Test.csv")
data3=pd.read_csv("Dataset2Train.csv")
data4=pd.read_csv("Dataset2Test.csv")

dataset=data1

X=dataset.iloc[:,0:-1].values
y=dataset.iloc[:,-1].values


##########################################################################Spliting train test
from sklearn.model_selection import train_test_split as TTS
X_train,X_test,y_train,y_test=TTS(X,y,test_size=0.7,random_state=0)

#####################################################



##########################################################################Stander Scaler
from sklearn.preprocessing import StandardScaler as SS
sc=SS()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

################################################################Trining Model Using Logistic Reg

from sklearn.linear_model import LogisticRegression as LogReg

Model=LogReg(random_state=0)
Model.fit(X_train,y_train)

y_pred_log=Model.predict(X_test)

print("Logistic Reg",end=" ")
Accuracy(y_test,y_pred_log)
############################################################# Trining Model Using K-NN
from sklearn.neighbors import KNeighborsClassifier as KNN

Model=KNN(n_neighbors=10,metric="minkowski",  p=2)
Model.fit(X_train,y_train)
y_pred_KNN=Model.predict(X_test)

print("K-NN",end=" ")
Accuracy(y_test, y_pred_KNN)
#################################################################rining Model Using Decision Tree
from sklearn.tree import DecisionTreeClassifier as DTC

Model=DTC(criterion='entropy',random_state=0)
Model.fit(X_train,y_train)

y_pred_Decision=Model.predict(X_test)

print("Decision Tree",end=" ")
Accuracy(y_test,y_pred_Decision)
##############################################
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)

Model.fit(X_train, y_train)

# Predicting the Test set results
y_pred_Rand = Model.predict(X_test)


print("Random Forest",end=" ")
Accuracy(y_test, y_pred_Rand)
