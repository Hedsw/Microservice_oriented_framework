
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from flask import redirect,send_from_directory

import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys
from abc import ABC, abstractmethod

app=Flask(__name__, template_folder='../templates')

class Strategy(ABC):
    @abstractmethod
    def evaluation(self):
        print("main, method called")
        pass

class Context():
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy
        df = pd.read_csv('../dataset/process_data.csv')
        y = df['fraud']
        X =  df.loc[:, df.columns != 'fraud']
        x_train, self.x_test, y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Logistic Regression
        print ("xxxxxxxxxxxxxxx Load Logistic Regression xxxxxxxxxxxxxxx")
        self.logreg = LogisticRegression(random_state=42)
        self.logreg.fit(x_train, y_train)
        logreg_y_pred = self.logreg.predict(self.x_test)
        self.logreg_cm = confusion_matrix(logreg_y_pred, self.y_test, labels=[1,0])

        # Random Forest
        print ("xxxxxxxxxxxxxxx Load Random Forest xxxxxxxxxxxxxxx")
        self.rf = RandomForestClassifier(random_state=42)
        self.rf.fit(x_train, y_train)
        y_pred = self.rf.predict(self.x_test)
        self.forest_cm = confusion_matrix(y_pred, self.y_test, labels=[1,0])

        # Support Vector Machine
        print ("xxxxxxxxxxxxxxx Load Support Vector Machine xxxxxxxxxxxxxxx")
        self.svc = SVC(random_state=42)
        self.svc.fit(x_train, y_train)
        svc_y_pred = self.svc.predict(self.x_test)
        self.svc_cm = confusion_matrix(svc_y_pred, self.y_test, labels=[1,0])

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def logic(self) -> None:
        print("===== Context: Calling evaluation on data using the strategy =====")
        result = self._strategy.evaluation(self.logreg_cm, self.forest_cm, self.svc_cm, self.logreg,self.rf,self.svc, self.x_test,self.y_test)
        print("Task:  ",",".join(result))

class ConcreteEvaluationHeatMap(Strategy):
    def evaluation(self, logreg_cm, forest_cm, svc_cm,logreg,rf,svc, x_test,y_test):
        print("=====   Concrete Evaluation HeatMap   =======")
        try:        
            plt.clf()
            sns.heatmap(logreg_cm, cmap="RdPu", annot=True, fmt=".0f",xticklabels = ["Fraudulent", "Legitimate"], yticklabels = ["Fraudulent", "Legitimate"])
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.title("Logistic Regression")
            plt.savefig("logistic_regression")

            plt.clf()
            sns.heatmap(forest_cm, cmap="RdPu", annot=True, fmt=".0f",xticklabels = ["Fraudulent", "Legitimate"], yticklabels = ["Fraudulent", "Legitimate"])
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.title("Random Forest")
            plt.savefig("random_forest")
            
            plt.clf()
            sns.heatmap(svc_cm, cmap="RdPu", annot=True, fmt=".0f",xticklabels = ["Fraudulent", "Legitimate"], yticklabels = ["Fraudulent", "Legitimate"])
            plt.ylabel("True class")
            plt.xlabel("Predicted class")
            plt.title("Support Vector Machine")
            plt.savefig("support_vector_machine")

            # return send_from_directory("./", filename="logistic_regression.png", as_attachment=True)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

class ConcreteEvaluationResult(Strategy):
    def evaluation(self, logreg_cm, forest_cm, svc_cm,logreg,rf,svc, x_test,y_test):
        print("=====   Concrete Evaluation Result   =======")
        print("\033[1m The result is telling us that we have: ",(logreg_cm[0,0]+logreg_cm[1,1]),"correct predictions\033[1m")
        print("\033[1m The result is telling us that we have: ",(logreg_cm[0,1]+logreg_cm[1,0]),"incorrect predictions\033[1m")
        print("\033[1m We have a total predictions of: ",(logreg_cm.sum()))
        print(classification_report(y_test, logreg.predict(x_test)))

        print("\033[1m The result is telling us that we have: ",(forest_cm[0,0]+forest_cm[1,1]),"correct predictions\033[1m")
        print("\033[1m The result is telling us that we have: ",(forest_cm[0,1]+forest_cm[1,0]),"incorrect predictions\033[1m")
        print("\033[1m We have a total predictions of: ",(forest_cm.sum()))
        print(classification_report(y_test, rf.predict(x_test)))

        print("\033[1m The result is telling us that we have: ",(svc_cm[0,0]+svc_cm[1,1]),"correct predictions\033[1m")
        print("\033[1m The result is telling us that we have: ",(svc_cm[0,1]+svc_cm[1,0]),"incorrect predictions\033[1m")
        print("\033[1m We have a total predictions of: ",(svc_cm.sum()))
        print(classification_report(y_test, svc.predict(x_test)))

class Evaluation:
    @app.route('/')
    @app.route('/index')
    def index():
        return render_template("index.html")

    @app.route('/api/evaluation', methods=['GET'], endpoint = 'evaluation')
    def evaluation():
        try:
            Print = request.args.get('print')
            HeatMap = request.args.get('heatmap')

            if HeatMap == "HeatMap":
                context = Context(ConcreteEvaluationHeatMap())
                print("===========================================")
                print("Client: Strategy is set to Concrete Evaluation HeatMap.")
                context.logic()

            if Print == "Print":
                context = Context(ConcreteEvaluationResult())
                print("===========================================")
                print("Client: Strategy is set to Concrete Evaluation Result.")
                context.logic()

            return redirect("http://localhost:5005/", code=302)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5005)
    print("Port Number: ", PORT)
    app.run(debug=True, host='0.0.0.0', port=PORT)
