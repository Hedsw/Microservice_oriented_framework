# from msilib.schema import Class
from abc import abstractmethod
import os
import pandas as pd

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

from abc import ABC, abstractmethod
from flask import request, jsonify, render_template, redirect
import sys

UPLOAD_FOLDER = '../dataset'

app=Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class Strategy(ABC):
    @abstractmethod
    def fileconvert(self):
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

    def logic(self) -> None:
        print("===========================================")
        self._strategy.datasetprocessor()
        print("===========================================")    

class FileUpload:
    @app.route('/')
    @app.route('/index')
    def index():
        return render_template("index.html")

    @app.route('/uploader', methods = ['POST', 'GET'], endpoint = 'upload')
    def upload():
        if request.method == 'POST':
            f = request.files['File']
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("===========================================")
            print("File saved successfully")
        return render_template("index.html")

class xlsx_fileupload:
    def __init__(self, file):
        self.file = file # Get Excel File name   
    def datasetprocessor():
        read_file = pd.read_excel(file) # Read the file name from HTML
        # CONVERT the EXCEL FILE to CSV 
        # AND STORE THE CONVERTED FILE TO dataset folder
        read_file.to_csv ("frauddetection.csv", 
                        index = None,
                        header=True)
            
        # read csv file and convert 
        # into a dataframe object
        # STORE THE FILE into dataset folder
        pd.DataFrame(pd.read_csv("frauddetection.csv"))
        
        # show the dataframe
        print("File Converter from excel to csv is finished")
        f = request.files['File']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("File saved successfully")
        return render_template("index.html")

class csv_conveter:
    def __init__(self, file):
        self.file = file # Get Excel File name   
    def datasetprocessor(self, file): #In file, file nmae will be described 
        # YOU NEED TO FIX THE CODE INSIDE HERE 
        print("CSV FILE CONVERTER")
        f = request.files['File']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("===========================================")
        print("File saved successfully")

@app.route('/api/fileUpload', methods=['GET'], endpoint = 'datasetupload')
def datasetupload():
    try:
        # IF THE DATASET File is ended to csv 
        # PLEASE CHECK THE Uploaded dataset Name and.. give it to
        # the if statement
        
        print("Aglorithms to run: ", CSV ,XLSX)
        # IF THE DATASET File is ended to csv 
        if CSV == "CSV":
            context = Context(xlsx_fileupload())
            print("===========================================")
            print("Client: Strategy is set to randomforest.")
            context.logic()
        # IF THE DATASET File is ended to XLSX 
        if XLSX == "XLSX":
            print("===========================================")
            print("Client: Strategy is set to SupportVectorMachine.")
            context =  Context(csv_conveter())
            context.logic()
        
        return redirect("http://localhost:5010/", code=302)
    except:
        e = sys.exc_info()[0]
        return jsonify({'error': str(e)})
   
# IF USER WANT TO ADD OTHER FILE TYPES, 
# They have to build another concrete component! 

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5009)
    print("Port Number: ", PORT)
    #supportvectormachine()
    app.run(debug=True, host='0.0.0.0', port=PORT)
