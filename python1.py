import pandas as pd
import numpy as np 
def Accuracy(y_test,y_pred):
    from sklearn.metrics import confusion_matrix as CM
    cm=CM(y_test,y_pred)
    degree=cm.sum()
    
    S=0
    for i in range(len(cm)):
        for j in range(len(cm)):
            if(i==j):
                S=S+cm[i][j]
    print("degree: ",degree)
    print("S: ",S)
    print("Accuracy: ",int((S/degree)*100),"%")


def concat(data1, data2):
    data8=list(np.array(data1))
    data9=list(np.array(data2))
    data8.extend(data9)

    data11=pd.DataFrame(data8)
    
    return data11

def Print(X):
    for i in X:
        print(i,":",int(X[i]*100),end=" |")
    print("\n----------")

def Precision(y_test, y_pred):
    from sklearn.metrics import confusion_matrix as CM
    cm=CM(y_test,y_pred)
    Dataset_size=cm.sum()
    TP=cm[0][0]
    FP=cm[0][1]
    
    TN=cm[1][1]
    FN=cm[1][0]
    
    precision= TP/(TP+FP)
    
    return precision

def Recall(y_test, y_pred):
    from sklearn.metrics import confusion_matrix as CM
    cm=CM(y_test,y_pred)
    Dataset_size=cm.sum()
    TP=cm[0][0]
    FP=cm[0][1]
    
    TN=cm[1][1]
    FN=cm[1][0]
    
    recall= TP/(TP+FN)
    
    return recall

def F1(precision,recall):
    f1=(2*precision*recall)/(precision+recall)
    return f1

def Accuracy2(y_test,y_pred):
    from sklearn.metrics import confusion_matrix as CM
    cm=CM(y_test,y_pred)
    Dataset_size=cm.sum()
    TP=cm[0][0]
    FP=cm[0][1]
    
    TN=cm[1][1]
    FN=cm[1][0]
    
    accuracy=(TP+TN)/(TP+FN+TN+FP)
    
    return accuracy

def Statistics(y_test,y_pred):
    precision=Precision(y_test, y_pred)
    recall=Recall(y_test,y_pred)
    
    f1=F1(precision,recall)
    accuracy=Accuracy2(y_test, y_pred)
    I="659810097108119971049798"
    return {"precision":precision,"recall":recall,"f1":f1,"accuracy":accuracy}

def RandomForestClassifier(data1,data2):
    
    X_train=data1.iloc[:,0:-1].values
    y_train=data1.iloc[:,-1].values
    X_test=data2.iloc[:,0:-1].values
    y_test=data2.iloc[:,-1].values
    
    from sklearn.preprocessing import StandardScaler as SS
    sc=SS()

    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)

    
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    X=Statistics(y_test,y_pred)
    
    Print(X)

def SVM_Algo(data1, data2):
    X_train=data1.iloc[:,0:-1].values
    y_train=data1.iloc[:,-1].values
    X_test=data2.iloc[:,0:-1].values
    y_test=data2.iloc[:,-1].values
    
    from sklearn.preprocessing import StandardScaler as SS
    sc=SS()

    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)
    
    from sklearn.svm import SVC
    svm_model = SVC(kernel='linear', random_state=0)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    
    X=Statistics(y_test,y_pred_svm)
    
    Print(X)
    
def KNN(data1,data2):
    
    X_train=data1.iloc[:,0:-1].values
    y_train=data1.iloc[:,-1].values
    X_test=data2.iloc[:,0:-1].values
    y_test=data2.iloc[:,-1].values
    
    from sklearn.preprocessing import StandardScaler as SS
    sc=SS()

    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)

    
    from sklearn.neighbors import KNeighborsClassifier as KNN

    Model=KNN(n_neighbors=10,metric="minkowski",  p=2)
    Model.fit(X_train,y_train)
    y_pred=Model.predict(X_test)
    
    X=Statistics(y_test,y_pred)
    
    Print(X)


def DecisionTreeClassifier(data1, data2):
    
    X_train=data1.iloc[:,0:-1].values
    y_train=data1.iloc[:,-1].values
    X_test=data2.iloc[:,0:-1].values
    y_test=data2.iloc[:,-1].values
    
    from sklearn.preprocessing import StandardScaler as SS
    sc=SS()

    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)

    ################################################################Trining Model Using Logistic Reg

    from sklearn.tree import DecisionTreeClassifier as DTC

    Model=DTC(criterion='entropy',random_state=0)
    Model.fit(X_train,y_train)

    y_pred=Model.predict(X_test)

    X=Statistics(y_test,y_pred)
    
    Print(X)

def splitDataset(data1, data2, size=50):
    data=concat(data1, data2)
    data_size=len(data)
    divide=100/size
    test_size=int(data_size/divide)
    train_size=data_size-test_size
    data1_new=data.iloc[:train_size,:]
    data2_new=data.iloc[train_size:,:]
    
    return [data1_new, data2_new]
    
    

def mainFun(test_size=20):
    data1=pd.read_csv("Dataset1Train.csv")
    data2=pd.read_csv("Dataset1Test.csv")

    D1=splitDataset(data1, data2, test_size)

    data1=D1[0]
    data2=D1[1]


    data3=pd.read_csv("Dataset2Train.csv")
    data4=pd.read_csv("Dataset2Test.csv")

    D2=splitDataset(data3, data4, test_size)

    data3=D2[0]
    data4=D2[1]




    data5=pd.read_csv("Dataset3Train.csv")
    data6=pd.read_csv("Dataset3Test.csv")

    D3=splitDataset(data5, data6, test_size)

    data5=D3[0]
    data6=D3[1]


    print("\n=============================DecisionTreeClassifier==============================\n")

    print("Aposemat IoT-23:: ")
    DecisionTreeClassifier(data1,data2)

    print("N-Baiot:: ")
    DecisionTreeClassifier(data3,data4)

    print("Bot-IoT:: ")
    DecisionTreeClassifier(data5,data6)



    print("\n==================================KNN============================================\n")
    print("Aposemat IoT-23:: ")
    KNN(data1,data2)
    print("N-Baiot:: ")
    KNN(data3,data4)
    print("Bot-IoT:: ")
    KNN(data5,data6)



    print("\n=========================RandomForestClassifier==================================\n")
    print("Aposemat IoT-23:: ")
    RandomForestClassifier(data1,data2)

    print("N-Baiot:: ")
    RandomForestClassifier(data3,data4)

    print("Bot-IoT:: ")
    RandomForestClassifier(data5,data6)



    print("\n=========================SVM==================================\n")
    print("Aposemat IoT-23:: ")
    SVM_Algo(data1,data2)

    print("N-Baiot:: ")
    SVM_Algo(data3,data4)

    print("Bot-IoT:: ")
    SVM_Algo(data5,data6)

print("------------------------train_data=80 test_data=20-------------------------------------")

mainFun(20)
print("------------------------train_data=50 test_data=50-------------------------------------")

mainFun(50)








