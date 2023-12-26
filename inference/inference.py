from flask import Flask, request, jsonify
import pickle , os , time , wandb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# Get the number of CPUs using os.cpu_count()
num_cpus = os.cpu_count()
print(f"num cpus: {num_cpus}")
app = Flask(__name__)

name_="mustafakeser"
project_="marketing-campaign-wb"
entity_=None
custom_date = os.environ["CUSTOM_DATE"]
run = wandb.init(
                project=project_, 
                entity=entity_, 
                   job_type="train",
                name = "TEST-API-"+custom_date,
                tags = ["TEST"]
                
    )

artifact = run.use_artifact(f'mustafakeser/marketing-campaign-wb/pipeline_fbtydk8g:v0', type='pipeline')
artifact_dir = artifact.download()

artifact2 = run.use_artifact(f'mustafakeser/marketing-campaign-wb/preprocessor_fbtydk8g:v0', type='preprocessor')
artifact_dir2 = artifact2.download()






# Load preprocessor and pipeline
with open(os.path.join(artifact_dir2,"preprocessor.pickle"), "rb") as file:
    loaded_prep = pickle.load(file)

with open(os.path.join(artifact_dir,"pipeline.pickle"), "rb") as file:
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

def run_server():
    # Run the server for 2 minutes
    app.run(port=5000, threaded=True)

def send_curl():
    print("Server is Started...\n")
    time.sleep(3)  # Adjust sleep duration if needed
    os.system("""curl -X POST -H "Content-Type: application/json" -d '{"age": 32, "job": "management", "marital": "married", "education": "tertiary", "default": "no", "balance": 1500, "housing": "no", "loan": "no", "contact": "cellular", "day": 28, "month": "jan", "duration": 458, "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown", "is_elderly": 1, "has_housing_loan": 0, "is_married": 1, "has_previous_contact": 1, "is_tertiary_educated": 1, "is_admin_job": 0, "has_default": 0, "is_month_may": 0, "is_loan": 0, "campaign_greater_than_1": 1}' http://127.0.0.1:5000/predict""")

    print("\nTest Successful")
    print("Server is being closed...")
    run.finish()
    os._exit(0)

if __name__ == '__main__':
    # Run the server and send curl command in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        server_future = executor.submit(run_server)
        
        time.sleep(1)  
        executor.submit(send_curl)
        
        time.sleep(2) 
        # Wait for the server to finish
        server_future.result()
        os._exit(0)
        
    os._exit(0)
os._exit(0)
    
    

#curl -X POST -H "Content-Type: application/json" -d '{"age": 32, "job": "management", "marital": "married", "education": "tertiary", "default": "no", "balance": 1500, "housing": "no", "loan": "no", "contact": "cellular", "day": 28, "month": "jan", "duration": 458, "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown", "is_elderly": 1, "has_housing_loan": 0, "is_married": 1, "has_previous_contact": 1, "is_tertiary_educated": 1, "is_admin_job": 0, "has_default": 0, "is_month_may": 0, "is_loan": 0, "campaign_greater_than_1": 1}' http://127.0.0.1:5000/predict
