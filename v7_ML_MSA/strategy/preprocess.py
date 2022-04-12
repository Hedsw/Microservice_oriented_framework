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
from sklearn.linear_model import Ridge


from typing import List
import requests
import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys

from abc import ABC, abstractmethod

app=Flask(__name__, template_folder='../templates')

class Component():
    """
    The base Component interface defines operations that can be altered by
    decorators.
    """

    def preprocess(self) -> str:
        pass


class ConcreteComponent(Component):
    """
    Concrete Components provide default implementations of the operations. There
    might be several variations of these classes.
    """

    def __init__(self) -> None:
        self.data_x1 = None
        self.df = pd.read_csv('../dataset/frauddetection.csv')

    def preprocess(self) -> str:
        print("================ Concrete - preprocess  ================")
        try:
            cols = list()
            for i in range (0, len(self.data_x1['Importance'])):
                if self.data_x1['Importance'][i] == 1:
                    cols.append(self.data_x1['Feature'][i])
            
            result = pd.concat([self.df[self.cols], self.df['fraud']], axis=1)
            result.to_csv('../dataset/process_data.csv', encoding='utf-8', index=False)
            return jsonify({'sucess': 200, "message":"preprocessing is task is done"})
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

class Decorator(Component): 
    """
    The base Decorator class follows the same interface as the other components.
    The primary purpose of this class is to define the wrapping interface for
    all concrete decorators. The default implementation of the wrapping code
    might include a field for storing a wrapped component and the means to
    initialize it.
    """

    _component: Component = None

    def __init__(self, component: Component) -> None:
        self._component = component

    @property
    def component(self) -> Component:
        """
        The Decorator delegates all work to the wrapped component.
        """

        return self._component

    def preprocess(self) -> str:
        return self._component.preprocess()


class ConcreteDecoratorLASSO(Decorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """
    # https://brownbears.tistory.com/557 이거 참조해서 다시 만들 것.. 지금 이거 틀렸음 
    
    def preprocess(self) -> str:
        """
        Decorators may call parent implementation of the operation, instead of
        calling the wrapped object directly. This approach simplifies extension
        of decorator classes.
        """
        return f"ConcreteDecoratorA({self.component.preprocess()})"
    
    def selectfeaturing(self):
        print("======   Concrete Component(Lasso)- select featuring starts     =========")
        try:
            df_vars = self.df.columns.values.tolist()
            y = ['fraud']
            X = [i for i in df_vars if i not in y]
            model = linear_model.Lasso(alpha=0.1)
            rfe = RFE(model)
            rfe = rfe.fit(self.df[X], self.df[y].values.ravel())
            self.data_x1 = pd.DataFrame({
            'Feature': self.df[X].columns,'Importance': rfe.ranking_},)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

class ConcreteDecoratorRidegeRegression(Decorator):
    """
    Decorators can execute their behavior either before or after the call to a
    wrapped object.
    """

    def preprocess(self) -> str:
        return f"ConcreteDecoratorB({self.component.preprocess()})"

    def selectfeaturing(self):
        print("======   Concrete Component(Ridege Regression) - select featuring starts     =========")
        try:
            df_vars = self.df.columns.values.tolist()
            y = ['fraud']
            X = [i for i in df_vars if i not in y]
            model = Ridge(alpha=1.0)

            rfe = RFE(model)
            rfe = rfe.fit(self.df[X], self.df[y].values.ravel())
            self.data_x1 = pd.DataFrame({
            'Feature': self.df[X].columns,'Importance': rfe.ranking_},)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})
class PreProcessing:
    @app.route('/')
    @app.route('/index')
    def index():
        return render_template("index.html")

    @app.route('/api/preprocess/', methods=['GET'], endpoint = 'preprocess')
    def preprocess():
        try:
            flag = request.args.get('process')
            print("flag",flag)
            concrete = ConcreteComponent()

            if flag == "Lasso":
                print("===========================================")
                print("Client: Strategy is set to rfe_lasso.")
                decorator_lasso = ConcreteDecoratorLASSO(concrete)
                decorator_lasso.preprocess()
                decorator_lasso.selectfeaturing()

            if flag == "RG":
                print("===========================================")
                print("Client: Strategy is set to selectfeaturing.")
                decorator_RG = ConcreteDecoratorRidegeRegression(concrete)
                decorator_RG.preprocess()
                decorator_RG.selectfeaturing()
            return redirect("http://localhost:5011/", code=302)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5011)
    app.run(debug=True, host='0.0.0.0', port=PORT)