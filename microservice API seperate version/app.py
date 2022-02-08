import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.svm import SVC

import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys

app=Flask(__name__, template_folder='templates')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/api/preprocess', methods=['GET'], endpoint = 'preprocess')
def preprocess():
    try:
        # Service 1 
        df = pd.read_csv('dataset/frauddetection.csv')
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
        result = result.iloc[: , 1:]
        print(result.head(1))

        
        result.to_csv('dataset/process_data.csv', encoding='utf-8', index=False)
        return jsonify({'sucess': 200, "message":"preprocessing is task is done"})
        # return render_template('index.html')

    except:
        e = sys.exc_info()[0]
        return jsonify({'error': str(e)})

@app.route('/api/randomforest', methods=['GET'], endpoint = 'randomforest')
def randomforest():
    try:
        # Service 2 
        df = pd.read_csv('dataset/process_data.csv')
        print(df)
        y = df['fraud']
        X =  df.loc[:, df.columns != 'fraud']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        """
        Different from Here 
        """
        logreg = LogisticRegression(random_state=42)
        logreg.fit(x_train, y_train)
        print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(x_test))))
        
        kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
        print("Kfold is ready")
        modelcv = RandomForestClassifier(random_state=42)
        print("Classifier is ready")
        scoring = 'accuracy'
        results = model_selection.cross_val_score(modelcv, x_train, y_train, cv = kfold, scoring = scoring)
        print("10-fold cross validation average accuracy of the random forest model: %.3f" % (results.mean()))
        return jsonify({'sucess': 200, "message":"randomforest is task is done", "cross validation average accuracy": results.mean()})
        # return render_template('index.html')
    except:
        e = sys.exc_info()[0]
        return jsonify({'error': str(e)})

@app.route('/api/svm', methods=['GET'], endpoint = 'svm')
def supportvectormachine():
    try:
        # Service 3
        df = pd.read_csv('dataset/process_data.csv')
        print(df)
        X =  df.loc[:, df.columns != 'fraud']
        y = df['fraud']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #X = X.values.reshape(-1,1)

        """
        Different from Here 
        """
        svc = SVC(random_state=42)
        svc.fit(x_train, y_train)
        print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(x_test))))

        kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
        print("Kfold is ready")
        modelCV = SVC(random_state=42)
        print("SVC is ready")
        scoring = 'accuracy'
        results = model_selection.cross_val_score(modelCV, x_train, y_train, cv = kfold, scoring = scoring)
        print("10-fold cross validation average accuracy of the support vector machine model: %.3f" % (results.mean()))
        return jsonify({'sucess': 200, "message":"svm is task is done", "cross validation average accuracy": results.mean()})
        # return render_template('index.html')
    except:
        e = sys.exc_info()[0]
        return jsonify({'error': str(e)})


    #return render_template("index.html")
if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5003)
    print("Port Number: ", PORT)
    app.run(debug=True, host='0.0.0.0', port=PORT)
