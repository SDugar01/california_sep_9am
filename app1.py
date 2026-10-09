from flask import Flask, request
import numpy as np
import sklearn
import joblib

obj=joblib.load('califor.joblib')
model=obj['Model']
col=obj['Columns']

app=Flask(__name__)
@app.route('/')
def main():
    return("Prediction")

@app.route('/predict')
def predict():
    input=[]
    for i in col:
        val=request.args.get(f'{i}',type=float)
        input.append(val)
    out=model.predict([input])
    app.logger.info(f"Prediction output: {out}")
    app.logger.info(type(out))
    return(out.tolist())

if __name__ == "__main__":
    app.run(debug=True)