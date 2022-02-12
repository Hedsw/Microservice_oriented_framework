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

app=Flask(__name__, template_folder='../templates')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/api/preprocess', methods=['GET'], endpoint = 'preprocess')
def preprocess():
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

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5004)
    print("Port Number: ", PORT)
    app.run(debug=True, host='0.0.0.0', port=PORT)
