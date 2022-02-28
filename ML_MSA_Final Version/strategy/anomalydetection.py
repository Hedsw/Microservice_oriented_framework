from cgitb import reset
from dataclasses import dataclass
from urllib import response
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from typing import List
import requests
import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys

from abc import ABC, abstractmethod

app=Flask(__name__, template_folder='../templates')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

class Strategy(ABC):
    @abstractmethod
    def trainml(self):
        pass
    
    @abstractmethod
    def anomalydetection(self):
        pass
    
class Context():
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def common_trainml(self) -> None:
        print("Select Featuring")
        self._strategy.trainml()        
        print("Select Featuring is done")
        
    def common_anomalydetection(self) -> None:
        print("Preprocess")
        self._strategy.anomalydetection()
        print("Preprocess is done")
        
class randomforest(Strategy):
    def trainml(self):
        print("Concrete Component(RandomForest) - trainML")
        
    def anomalydetection(self):
        print("Concrete Component(RandomForest) - AnomalyDetection")
        try:
            df = pd.read_csv('../dataset/process_data.csv')
            print(df)
            y = df['fraud']
            X =  df.loc[:, df.columns != 'fraud']
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            
            """
            Different from Here 
            """
            logreg = LogisticRegression(random_state=42)
            logreg.fit(x_train, y_train)
            print("===========================================")
            print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(x_test))))
            kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
            print("===========================================")
            print("Kfold is ready")
            modelcv = RandomForestClassifier(random_state=42)
            print("===========================================")
            print("Classifier is ready")
            scoring = 'accuracy'
            results = model_selection.cross_val_score(modelcv, x_train, y_train, cv = kfold, scoring = scoring)
            print("===========================================")
            print("10-fold cross validation average accuracy of the random forest model: %.3f" % (results.mean()))
            return jsonify({'sucess': 200, "message":"randomforest is task is done", "cross validation average accuracy": results.mean()})
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

class supportvectormachine(Strategy):
    def trainml(self):
        print("Concrete Component(SupportVectorMachine) - trainML")
        
    def anomalydetection(self):
        print("Concrete Component(SupportVectorMachine) - AnomalyDetection")
        try:
            # Service 3
            df = pd.read_csv('../dataset/process_data.csv')
            # print(df)
            X =  df.loc[:, df.columns != 'fraud']
            y = df['fraud']

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            svc = SVC(random_state=42)
            svc.fit(x_train, y_train)
            print("===========================================")
            print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(x_test))))

            kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
            print("===========================================")
            print("Kfold is ready")
            modelCV = SVC(random_state=42)
            print("===========================================")
            print("SVC is ready")
            scoring = 'accuracy'
            results = model_selection.cross_val_score(modelCV, x_train, y_train, cv = kfold, scoring = scoring)
            print("===========================================")
            print("10-fold cross validation average accuracy of the support vector machine model: %.3f" % (results.mean()))
            return jsonify({'sucess': 200, "message":"svm is task is done", "cross validation average accuracy": results.mean()})
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

class logisticregression(Strategy):
    def trainml(self):
        print("Concrete Component(LogisticRegression) - trainML")
        
    def anomalydetection(self):
        print("Concrete Component(LogisticRegression) - AnomalyDetection")
        try:
            print("Have to build Logistic Regression")
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5010)
    app.run(debug=True, host='0.0.0.0', port=PORT)