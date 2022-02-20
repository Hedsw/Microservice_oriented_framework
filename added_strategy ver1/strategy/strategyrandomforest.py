from cgitb import reset
from urllib import response
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
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
    def api_call(self, data):
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

    def business_logic(self) -> None:
        print("Context: Calling API on data using the strategy")
        data = pd.read_csv('../dataset/process_data.csv')
        result = self._strategy.api_call(data)
        print("done ",",".join(result))

class ConcretePreprocessing(Strategy):
    def api_call(self, data):
        response = requests.get('http://localhost:5001/api/preprocess')
        print(response.status_code)
        return data

class ConcreteRandomForest(Strategy):
    def api_call(self, data):
        response =  requests.get('http://localhost:5002/api/randomforest')
        print("status ",response.status_code)
        return data

class ConcreteSVM(Strategy):
    def api_call(self, data):
        response = requests.get('http://localhost:5003/api/svm')
        print("status ",response.status_code)
        return data

class ConcreteEvaluation(Strategy):
    def api_call(self, data):
        response = requests.get('http://localhost:5005/api/evaluation')
        return data

@app.route('/api/strategy', methods=['GET'], endpoint = 'randomforestapi')
def randomforestapi():
    try:
        context = Context(ConcretePreprocessing())
        print("===========================================")
        print("Client: Strategy is set to Preprocessing.")
        context.business_logic()
        print("===========================================")
        print("Client: Strategy is set to reverse sorting.")
        context.strategy = ConcreteRandomForest()
        context.business_logic()
        print("===========================================")
        print("Client: Strategy is set to RandomForest.")
        context.strategy = ConcreteSVM()
        context.business_logic()
        print("===========================================")
        print("Client: Strategy is set to Evaluation.")
        context.strategy = ConcreteEvaluation()
        context.business_logic()
    except:
        e = sys.exc_info()[0]
        return jsonify({'error': str(e)})

    return render_template("index.html")

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5000)
    app.run(debug=True, host='0.0.0.0', port=PORT)