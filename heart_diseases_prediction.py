'''
Dataset legend:

age
sex
cp->chest pain
trestbps-> resting blood pressure
chol-> cholesterol
fbs-> fasting blood sugar
restecg-> resting electrocardiographic results
thalach-> max heart rate during stress
exang-> excercise induced angina 
oldpeak-> ST depression induced by excercise relative to rest
slope-> slope of peak excercise ST segment
ca-> 
thal-> Thalassemia 1-3 based on severity
target-> 1:heart disease, 0:no issues
'''

import numpy as np;
import pandas as pd;
import warnings

#importing from scikit learn the functions for spliting data, logistic Rgression and accuracy score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore",category=UserWarning)

#opening the csv file into pandas dataframe
data=pd.read_csv(r'dataset\heart_disease_data.csv')

def getInput():
    list1=[0,0,0,0,0,0,0,0,0,0,0,0,0]
    list1[0]=int(input("Enter the age:"))
    list1[1]=int(input("Enter the sex:"))
    list1[2]=int(input("Enter the chest pain level (cp):"))
    list1[3]=int(input("Enter the resting blood pressure (trestbps):"))
    list1[4]=int(input("Enter the cholesterol level (chol):"))
    list1[5]=int(input("Enter the fasting blood sugar (fbs):"))
    list1[6]=int(input("Enter the resting ecg results (restecg):"))
    list1[7]=int(input("Enter the thalach:"))
    list1[8]=int(input("Enter the exang:"))
    list1[9]=float(input("Enter the oldpeak:"))
    list1[10]=int(input("Enter the slope:"))
    list1[11]=int(input("Enter the ca:"))
    list1[12]=int(input("Enter the Thalassemia (thal):"))
    tuple1=tuple(list1)
    return tuple1

#Specifying the target and the features
#set axis=0 if you are dropping a row
features=data.drop(columns='target',axis=1)
target=data['target']

#test_size=0.2 allocates 1/5th of the dataset as testing data to check accuracy of the model
#stratify=target ensures that there is an even distribution of 1s and 0s (heart condition) in the test data
#random_state is like a seed, using the same random_state will yeild the same results each time
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=1)


#defining the model
model=LogisticRegression()
#training the model
#model.fit(), tries to draw a relation between the parameters passed to it.
model.fit(features_train, target_train)
print("Training done!")


#getting the accuracy scores by comparing with target_test and target_train
features_train_prediction=model.predict(features_train)
features_test_prediction=model.predict(features_test)
training_accuracy=accuracy_score(features_train_prediction,target_train)
testing_accuracy=accuracy_score(features_test_prediction,target_test)
print("Training Data Accuracy Score:", training_accuracy)
print("Testing Data Accuracy Score:", testing_accuracy)

#getting the input
input=getInput()

#converting list to numpy array so that we can reshape it to a 1 row datapoint
#npinput is a one dimensional array of size (13,)
npinput=np.asarray(input)
#npinput is a two dimensional array with one row and 13 columns(columns can vary, indicated by -1) (1,13)
npinput_reshape=npinput.reshape(1, -1)

#predicting the output, returns a list with 1 element
prediction=model.predict(npinput_reshape)

if(prediction[0]==1):
    print("RISK OF HEART DISEASES")
elif(prediction[0]==0):
    print("NO RISK OF HEART DISEASES")
