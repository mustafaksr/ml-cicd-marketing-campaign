from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load preprocessor and pipeline
with open(os.path.join(os.getcwd(),"preprocessor.pickle"), "rb") as file:
    loaded_prep = pickle.load(file)

with open(os.path.join(os.getcwd(),"pipeline.pickle"), "rb") as file:
    loaded_pipeline = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from JSON request
        input_data = request.get_json()

        # Create a DataFrame from input data
        input_df = pd.DataFrame([input_data])

        # Perform pre-processing
        preprocessed_data = loaded_prep.transform(input_df)

        # Make predictions
        predictions = loaded_pipeline.predict_proba(preprocessed_data)[:, 1]

        # Return the prediction result
        return jsonify({'prediction': predictions[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)


#curl -X POST -H "Content-Type: application/json" -d '{"age": 32, "job": "management", "marital": "married", "education": "tertiary", "default": "no", "balance": 1500, "housing": "no", "loan": "no", "contact": "cellular", "day": 28, "month": "jan", "duration": 458, "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown", "is_elderly": 1, "has_housing_loan": 0, "is_married": 1, "has_previous_contact": 1, "is_tertiary_educated": 1, "is_admin_job": 0, "has_default": 0, "is_month_may": 0, "is_loan": 0, "campaign_greater_than_1": 1}' http://127.0.0.1:5000/predict
