import numpy as np
from flask import Flask,request,jsonify,render_template
import joblib
import random
import requests
import json

API_KEY = "lXz0XO1eh1nLZZLQ2m7wahlFW812sFxvI5d80PVEQAz9"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
app=Flask(__name__)
model=joblib.load("engine_model.sav")
@app.route('/')
def predict():
    return render_template('index.html')
@app.route('/y_predict', methods=['POST'])
def y_predict():
    x_test=[[int(x) for x in request.form.values()]]
    print(x_test)
    payload_scoring = {"input_data": [{"field": [['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']] , "values": [[ 2,  1, 4,  0,
        1,  5,  6,  1,
        1,  1,  2,  5,
        2,  9,  1,  4,
        5,  2,  8,  8,
        3,  3,  2,  1,
        3,  2,  4]]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/2d466513-0d23-4649-9345-753d51f7873d/predictions?version=2022-11-17', json=payload_scoring,
     headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")

    predictions = response_scoring.json()

    pred=predictions["predictions"][0]['values'][0][0]

    if( pred == 0):
        pred="No Failure Expected within 30 days."
    else:
        pred="Maintenance Required!! Expected a failure within 30 days."
    return render_template('index.html',ans=pred)
if (__name__=='__main__'):
    app.run(debug=False)