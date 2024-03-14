from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, template_folder='templates')

model = joblib.load('Model.pkl')  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():

    eeg_id = float(request.form.get('eeg_id', 0))
    seizure_vote = float(request.form.get('seizure_vote', 0))
    lpd_vote = float(request.form.get('lpd_vote', 0))
    gpd_vote = float(request.form.get('gpd_vote', 0))
    lrda_vote = float(request.form.get('lrda_vote', 0))
    grda_vote = float(request.form.get('grda_vote', 0))
    other_vote = float(request.form.get('other_vote', 0))

    #print(f"Received input: eeg_id={eeg_id}, seizure_vote={seizure_vote}, lpd_vote={lpd_vote}, gpd_vote={gpd_vote}, lrda_vote={lrda_vote}, grda_vote={grda_vote}, other_vote={other_vote}")

    new_data = pd.DataFrame({
        'eeg_id': [eeg_id],
        'seizure_vote': [seizure_vote],
        'lpd_vote': [lpd_vote],
        'gpd_vote': [gpd_vote],
        'lrda_vote': [lrda_vote],
        'grda_vote': [grda_vote],
        'other_vote': [other_vote]
    })

    #print("Input data for prediction:", new_data)

    predicted = model.predict(new_data)
    #print("Predicted probabilities:", predicted)

    index = np.argmax(predicted)
    classes = ['Seizure', 'GPD', 'LRDA', 'GRDA', 'LPD', 'Other']
    predicted_class = classes[index]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
