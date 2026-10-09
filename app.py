from flask import Flask, request, render_template
import sklearn
import numpy as np
import joblib

# Load the trained model
obj = joblib.load('califor.joblib')

model = obj['Model']
columns = obj['Columns']

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html', columns=columns)


@app.route('/predict', methods=['POST'])
def predict():

    Input = []

    for i in columns:
        val = request.form.get(i, type=float)
        Input.append(val)

    prediction = model.predict([Input])

    return render_template(
        'index.html',
        columns=columns,
        prediction=prediction[0]
    )


if __name__ == "__main__":
    app.run(debug=True)
