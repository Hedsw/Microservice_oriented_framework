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
        print("Task:  ",",".join(result))

class ConcretePreprocessingOne(Strategy):
    def api_call(self, data):
        data =  data.to_string()
        response = requests.get('http://localhost:5001/api/preprocess/preprocess1')
        return "Preprocessing is Done"

class ConcretePreprocessingTwo(Strategy):
    def api_call(self, data):
        response = requests.get('http://localhost:5001/api/preprocess/preprocess2')
        return "Preprocessing is Done"

class ConcreteRandomForest(Strategy):
    def api_call(self, data):
        response =  requests.get('http://localhost:5002/api/randomforest')
        return "Random Forest is done."

class ConcreteSVM(Strategy):
    def api_call(self, data):
        response = requests.get('http://localhost:5003/api/svm')
        return "SVM is done."

class ConcreteEvaluation(Strategy):
    def api_call(self, data):
        response = requests.get('http://localhost:5005/api/evaluation')
        return "Evaluation is done."

class ConcreteFileUpload(Strategy):
    def api_call(self, data):
        response = requests.get('http://localhost:5009/upload')
        return "File Upload is done."


@app.route('/api/strategy/randomforest', methods=['GET'], endpoint = 'randomforest')
def randomforest_api_stragtegy():
    try:
        context = Context(ConcretePreprocessingOne())
        print("===========================================")
        print("Client: Strategy is set to Preprocessing.")
        context.business_logic()
        print("===========================================")
        print("Client: Strategy is set to Random Forest.")
        context.strategy = ConcreteRandomForest()
        context.business_logic()
        print("===========================================")
        print("Client: Strategy is set to Evaluation.")
        context.strategy = ConcreteEvaluation()
        context.business_logic()
    except:
        e = sys.exc_info()[0]
        return jsonify({'error': str(e)})

    return render_template("index.html")


@app.route('/api/stragtegy/svm', methods=['GET'], endpoint = 'svm')
def svm_api_stragtegy():
    try:
        context = Context(ConcretePreprocessingTwo())
        print("===========================================")
        print("Client: Strategy is set to Preprocessing.")
        context.business_logic()
        print("===========================================")
        print("Client: Strategy is set to Support Vector Machine.")
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