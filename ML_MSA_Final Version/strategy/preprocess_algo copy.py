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
    def selectfeaturing(self, data):
        pass
    
    @abstractmethod
    def preprocess(self, data):
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

    def common_selectfeaturing(self) -> None:
        print("Select Featuring")
        self._strategy.selectfeaturing()        
        print("Select Featuring is done")
        
    def common_preprocess(self) -> None:
        print("Preprocess")
        self._strategy.preprocess()
        print("Preprocess is done")
        
class rfe_lasso(Strategy):
    def selectfeaturing(self, data):
        print("Concrete Component(Lasso)- select featuring starts")
        try:
            df = pd.read_csv('../dataset/frauddetection.csv')
        
            df_vars = df.columns.values.tolist()
            y = ['fraud']
            X = [i for i in df_vars if i not in y]
            model = linear_model.Lasso(alpha=0.1)

            rfe = RFE(model)
            rfe = rfe.fit(df[X], df[y].values.ravel())

            data_x1 = pd.DataFrame({
            'Feature': df[X].columns,'Importance': rfe.ranking_},)
            cols = []
            for i in range (0, len(data_x1['Importance'])):
                if data_x1['Importance'][i] == 1:
                    cols.append(data_x1['Feature'][i])
            print(cols)
            result = pd.concat([df[cols], df['fraud']], axis=1)
            result.to_csv('../dataset/process_data.csv', encoding='utf-8', index=False)
            return jsonify({'sucess': 200, "message":"preprocessing is task is done"})
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})
        
    def preprocess(self, data):
        print("Concrete Component(Lasso) - preprocess")

class rfe_logisticregression(Strategy): # Medium?????? RFE??? Logistic Regression??? ????????? ?????????
    def selectfeaturing(self, data):
        print("Concrete Component(LinearRegression) - select featuring starts")
        try:
            # Service 1 
            df = pd.read_csv('../dataset/frauddetection.csv')
            print(df)
        
            df_vars = df.columns.values.tolist()
            y = ['fraud']
            X = [i for i in df_vars if i not in y]

            model = LogisticRegression(solver='lbfgs', max_iter=3000)

            rfe = RFE(model)
            rfe = rfe.fit(df[X], df[y].values.ravel())

            data_x1 = pd.DataFrame({
            'Feature': df[X].columns,'Importance': rfe.ranking_},)
            
            cols = []
            for i in range (0, len(data_x1['Importance'])):
                if data_x1['Importance'][i] == 1:
                    cols.append(data_x1['Feature'][i])
            print(cols)
            result = pd.concat([df[cols], df['fraud']], axis=1)
            result.to_csv('../dataset/process_data.csv', encoding='utf-8', index=False)
            return jsonify({'sucess': 200, "message":"preprocessing is task is done"})
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})
        
    def preprocess(self, data):
        print("Concrete Component(LinearRegression) - Preprocess")
        
if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5011)
    app.run(debug=True, host='0.0.0.0', port=PORT)