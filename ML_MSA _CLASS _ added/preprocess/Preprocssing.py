import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model


import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys

app=Flask(__name__, template_folder='../templates')

class Upload:
    @app.route('/')
    @app.route('/index')
    def index():
        return render_template("index.html")

    @app.route('/api/preprocess/preprocess1', methods=['GET'], endpoint = 'preprocessOne')
    def preprocessOne():
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

    @app.route('/api/preprocess/preprocess2', methods=['GET'], endpoint = 'preprocessTwo')
    def preprocessTwo():
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
    PORT = os.environ.get('PORT', 5001)
    print("Port Number: ", PORT)
    app.run(debug=True, host='0.0.0.0', port=PORT)
