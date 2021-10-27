# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:51:38 2020

@author: Harikrishnan V
"""

import numpy as np
import pickle
from flask import Flask,render_template,request

app = Flask(__name__)

model = pickle.load(open("Kmodel.pkl","rb"))



@app.route("/")

def home():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])
def predict():
    size = [float(x) for x in request.form.values()]
    size = np.reshape(size,(-1,4))
    
    prediction = model.predict(size)
    prediction = prediction.item()
    
    if prediction == 0:
        out="iris-setosa"
    elif prediction == 1:
        out ="iris-versicolor"
    else:
        out="iris-virginica"
    return render_template("result.html",Results= "The species is :{}".format(out))
    
if __name__ == "__main__":
    app.run()
    
    
    


