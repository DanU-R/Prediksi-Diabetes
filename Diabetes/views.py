from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

def home(request):
    return render(request, 'home.html')

def awal(request):
    return render(request, 'awal.html')

def output(request):
    df = pd.read_csv('diabetes.csv')

    x = df.drop(columns='Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    scaler.fit(x)
    standar = scaler.transform(x)

    x = standar
    y = df['Outcome']
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
    
    model = svm.SVC(kernel='linear')
    model.fit(x_train, y_train)
    
    var1 = float(request.GET['Pregnancies'])
    var2 = float(request.GET['Glucose'])
    var3 = float(request.GET['BloodPressure'])
    var4 = float(request.GET['SkinThickness'])
    var5 = float(request.GET['Insulin'])
    var6 = float(request.GET['BMI'])
    var7 = float(request.GET['DiabetesPedigreeFunction'])
    var8 = float(request.GET['Age'])

    predict = [[var1, var2, var3, var4, var5, var6, var7, var8]]
    
    data = scaler.transform(predict)
    
    pred = model.predict(data)

    if pred == [1]:
        pred = "Anda Terkena Diabetes"
    else :
        pred = "Anda Tidak Terkena Diabetes"
        
    return render(request, 'home.html', {'hasil': pred})

