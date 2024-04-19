import pickle
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('Shipment Data.csv')
xgboost_model = pickle.load(open("main.pkl", 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    Mode = int(request.form.get('mode'))
    Distance = int(request.form.get('distance'))


    Weight = int(request.form.get('weight'))

    print(Distance, Mode, Weight)
    input = pd.DataFrame([[Distance, Mode, Weight]], columns=['Distance', 'Mode', 'Weight'])
    prediction = xgboost_model.predict(input)[0]


    return str(prediction)




if __name__ == '__main__':
    app.run(debug=True, port=5001)
