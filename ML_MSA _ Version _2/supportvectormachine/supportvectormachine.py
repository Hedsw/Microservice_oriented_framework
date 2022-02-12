import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

import os

from flask import Flask
from flask import request, jsonify, render_template, redirect
import sys

app=Flask(__name__, template_folder='../templates')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/api/svm', methods=['GET'], endpoint = 'svm')
def supportvectormachine():
    try:
        # Service 3
        df = pd.read_csv('../dataset/process_data.csv')
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
    except:
        e = sys.exc_info()[0]
        return jsonify({'error': str(e)})

#supportvectormachine()
if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5001)
    print("Port Number: ", PORT)
    #supportvectormachine()
    app.run(debug=True, host='0.0.0.0', port=PORT)
