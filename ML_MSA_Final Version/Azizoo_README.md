# microservice_ml

Service #1 (http://0.0.0.0:5010/api/anomalydetection)
1. Build AnomalyDetection.py 
    - RandomForest 
    - SupportVectorMachine
    - LogisticRegression

Service #2 (http://0.0.0.0:5011/api/preprocess)
2. Build Preprocess.py
    - Lasso
    - LinearRegression

Service #3 (http://0.0.0.0:5012/api/datasetupload)
3. fileupload.py 
    - It's already built. You don't have to build this. leave it! 

Service #4 (http://0.0.0.0:5012/api/datasetupload)
4. evaluation.py
    - It's already built. You don't have to build this. leave it! 

Description:

Same The application will work same steps with your version. (File Upload -> Preprocess -> ML algorithm -> Evaluation). However, at this time, We will give options to user and now we just have 4 services. 

When you open Index.html(template/index.html) on browser, you can see Dateset Upload, Preprocess, Anomaly Detection, Evaluation. Dataset upload is already built so you don't have to do nothing.

For Proprocess, 
When User click a checkbox and then submit button on Preprocess (Lasso or Logistic Regression), one of preprocess will be implemented. NO API. JUST call internally. (You can see the code on preprocess.py)

HOWEVER, You have to separate the preprocess function to two functions(feature selection and preprocess) So, when you call one of preprocesses(Lasso or Logistic Regression), you have to call like this: feature selection and then preprocess. 

For Anomaly Detection(Machine Learning Algo),
When User click a check box and then submit button on Anomaly detection (RandomForest, SVM or Logistic Regression), One of Anomaly Detection(RandomForest, SVM, or Logistic Regression) will be implemented. NO API. JUST call internally. (You can see the code on anomalydetection.py)

HOWEVER, you have to separate the one funtionality to two functionalties trainml and anomaly detection. In the Trainml function, train the ML. In the anomaly detection, run the ML.
So that when we call a ML algorithm, you have to call trainML() and then Anomaly Detection(). (To work like one functionality)

Secondly For the Anomaly Detection, When user click two checkbox and click submit button,
The two chosen ML algo will be run concurrently. (Don't worry about priority either one is fine) For example, I chose RamdonForest and SVM. And I clicked Submit button, then.. The both of RandomForest and SVM will be run.


