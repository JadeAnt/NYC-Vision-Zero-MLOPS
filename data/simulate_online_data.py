import os
import time
import pandas as pd
import requests

INFERENCE_URL = "http://fastapi_server:8000/predict"
REQUEST_DELAY = 3

def load_mock_prod_data(csv_path):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def make_inference_request(location_string):
    """Make a request to the inference endpoint."""
    try:
        response = requests.post(
            INFERENCE_URL,
            json={"intersection_id": location_string}
        )
        response.raise_for_status()

        return response.json()
    except Exception as e:
        print(f"Error making inference request for '{location_string}': {e}")
        return None

def simulate_production():
    """Main function to simulate production requests."""
    csv_path = "/mnt/object/production/simulated_production.csv"
    
    df = load_mock_prod_data(csv_path)
    if df is None:
        return
    
    # Process each row
    for index, row in df.iterrows():
        # Create concatenated intersection id
        location_string = f"{row['Roadway Name']}_{row['From']}"
        
        # Make inference request
        result = make_inference_request(location_string)
        
        if result:
            print(f"Result for {location_string}: {result}")
        
        time.sleep(REQUEST_DELAY)
        
    print("Simulation completed")

if __name__ == "__main__":
    simulate_production()
