from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('Knearforfruits.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Weight=float(request.form['Weight'])
        Sphericity=float(request.form['Sphericity'])
        prediction=model.predict([[Sphericity,Weight]])
        output=prediction[0]
        if output=='Orange':
            return render_template('index.html',prediction_text="This is Orange")  
        else:
            return render_template('index.html',prediction_text="This is Apple")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

