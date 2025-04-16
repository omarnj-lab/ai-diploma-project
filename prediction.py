import pandas as pd
import cudf
import joblib
from cuml.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def make_prediction(input_data):
    # Load the saved model and scaler
    loaded_model = joblib.load('lr_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')

    # Convert input data to a DataFrame (if it's not already)
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)

    # Convert pandas DataFrame to cuDF DataFrame
    input_data_cudf = cudf.DataFrame.from_pandas(input_data)

    # Scale the input data using the loaded scaler
    input_data_scaled = loaded_scaler.transform(input_data_cudf)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data_scaled)

    # You might want to load y_test here if you need to calculate accuracy
    y_test = pd.read_csv('y_test.csv')
    accuracy = accuracy_score(y_test, predictions.to_pandas())
    print(f"Model Accuracy: {accuracy:.4f}")

    print("Predictions:", predictions.to_pandas()) # Convert to pandas for printing
    return predictions.to_pandas() #Return as pandas Series
