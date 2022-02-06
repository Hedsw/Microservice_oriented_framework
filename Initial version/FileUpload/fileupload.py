import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage


UPLOAD_FOLDER = '../dataset'

app=Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")
	
@app.route('/uploader', methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['File']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("File saved successfully")
        return render_template("index.html")

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5004)
    print("Port Number: ", PORT)
    #supportvectormachine()
    app.run(debug=True, host='0.0.0.0', port=PORT)
