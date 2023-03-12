from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('UI.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(np.reshape(final_features, (1, 8)))

    output = round(prediction[0], 2)

    return render_template('UI.html', prediction_text='Predicted policy price is RS {}'.format(output))


if __name__ == "__main__":
    app.run()
