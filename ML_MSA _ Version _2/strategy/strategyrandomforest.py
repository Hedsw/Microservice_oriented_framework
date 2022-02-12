import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

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

class Context():
    #strategy: AbstractDownloader 
    def __init__(self, strategy):
        self._strategy = strategy 
    
    def dataset_interface(self):
        print("Context Interface")
        return self._strategy.dataset()

    def datapreperation_interface(self, test):
        print("datapreperation Interface")
        self._strategy.datapreperation(test)

class Strategy(ABC):
    @abstractmethod
    def datapreperation(self, test):
        pass
    
    @abstractmethod
    def modeling(self):
        pass
    
    @abstractmethod
    def dataset(self):
        pass

# Concrete Strategies
class randomforest_concrete(Strategy):
    def dataset(self):
        self.df = pd.read_csv('/Users/yunhyeoklee/Desktop/ML_MSA/dataset/frauddetection.csv')
        self.df.columns.values
        #print(self.df)
        return self.df
    
    def datapreperation(self, test):
        print("Real?")
        print(test)
        
    def modeling(self):
        pass

# Don't have to be a class   
@app.route('/api/randomforest', methods=['GET'], endpoint = 'randomforestapi')
def randomforestapi():
    try:
        """ 
        Data Collection
        """
        concrete_starategy = randomforest_concrete()
        context = Context(concrete_starategy)
        df = context.dataset_interface()
        #print(df)
        
        # 이거 파라메터 어떻게 넣냐..? 아놔..
        # It's a common logic
        # INPUT: df_vars , OUTPUT = cols 
        # context.datapreperation("Test")
        # From ..
        df_vars = df.columns.values.tolist()
        y = ['fraud']
        X = [i for i in df_vars if i not in y]

        model = LogisticRegression(solver='lbfgs', max_iter=3000)

        rfe = RFE(model)
        rfe = rfe.fit(df[X], df[y].values.ravel())

        data_x1 = pd.DataFrame({
        'Feature': df[X].columns,'Importance': rfe.ranking_},)
        
        cols = []
        # To Here.. 
        
        for i in range (0, len(data_x1['Importance'])):
            if data_x1['Importance'][i] == 1:
                cols.append(data_x1['Feature'][i])
        
        X = df[cols]
        y = df['fraud']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #X = X.values.reshape(-1,1)

        """
        Different from Here 
        """
        logreg = LogisticRegression(random_state=42)
        logreg.fit(x_train, y_train)
        print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(x_test))))
        
        kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
        print("Kfold is ready")
        modelCV = RandomForestClassifier(random_state=42)
        print("Classifier is ready")
        scoring = 'accuracy'
        results = model_selection.cross_val_score(modelCV, x_train, y_train, cv = kfold, scoring = scoring)
        print("10-fold cross validation average accuracy of the random forest model: %.3f" % (results.mean()))
    except:
        e = sys.exc_info()[0]
        return jsonify({'error': str(e)})

    return render_template("index.html")

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5002)
    #randomforestapi()
    print("Port Number: ", PORT)
    app.run(debug=True, host='0.0.0.0', port=PORT)
