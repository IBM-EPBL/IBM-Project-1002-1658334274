from flask import Flask, render_template, request
import numpy as np

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

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/y_predict', methods= ['POST'])
def  y_predict():
            l=[]
            l.append(float(request.form['id']))
            l.append(float(request.form['cycle']))
            l.append(float(request.form['setting1']))
            l.append(float(request.form['setting2']))
            l.append(float(request.form['setting3']))
            l.append(float(request.form['s1']))
            l.append(float(request.form['s2']))
            l.append(float(request.form['s3']))
            l.append(float(request.form['s4']))
            l.append(float(request.form['s5']))
            l.append(float(request.form['s6']))
            l.append(float(request.form['s7']))
            l.append(float(request.form['s8']))
            l.append(float(request.form['s9']))
            l.append(float(request.form['s10']))
            l.append(float(request.form['s11']))
            l.append(float(request.form['s12']))
            l.append(float(request.form['s13']))
            l.append(float(request.form['s14']))
            l.append(float(request.form['s15']))
            l.append(float(request.form['s16']))
            l.append(float(request.form['s17']))
            l.append(float(request.form['s18']))
            l.append(float(request.form['s19']))
            l.append(float(request.form['s20']))
            l.append(float(request.form['s21']))
            l.append(float(request.form['ttf']))
        
            print(l)
            # NOTE: manually define and pass the array(s) of values to be scored in the next line
            payload_scoring = {"input_data": [{"field":['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','ttf'], "values": [l]}]}

            response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/2d466513-0d23-4649-9345-753d51f7873d/predictions?version=2022-11-17', json=payload_scoring,
            headers={'Authorization': 'Bearer ' + mltoken})
            print("Scoring response")
            print(response_scoring.json())
            pred = response_scoring.json()
            output = pred['predictions'][0]['values'][0][0]
            print(output)

            if output >=1 and output <=2 :
                output="No Failure Expected within 30 days."
            else :
               output="Maintenance Required!! Expected a failure within 30 days."
            return render_template('index.html',ans=output)
if (__name__=='__main__'):
    app.run(debug=False)