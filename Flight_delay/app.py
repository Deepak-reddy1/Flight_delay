# Note: This code is a Flask web application that predicts flight delays using a pre-trained Random Forest model.
# It includes a web interface for users to input flight details and receive predictions.
# It also displays previous predictions made during the session.
# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle  # For loading the trained model

app = Flask(__name__)

# Load the pre-trained model 
rf_model = pickle.load(open('rf_model.pkl', 'rb'))  # Loading the RandomForest model

# Carrier mapping: Map carrier codes to numeric values used by the trained model
carrier_mapping = {
    '9E': 0,  # Endeavor Air
    'AA': 1,  # American Airlines
    'DL': 2,  # Delta Airlines
    'UA': 3,  # United Airlines
    'SW': 4,  # Southwest Airlines
    }

# samples of previous predictions data 
previous_predictions = [
    {'year': 2023, 'month': 8, 'carrier': 'Endeavor Air', 'predicted_delay': 15.30},
    {'year': 2023, 'month': 7, 'carrier': 'American Airlines', 'predicted_delay': 25.50},
    {'year': 2023, 'month': 6, 'carrier': 'Delta Airlines', 'predicted_delay': 10.20},
    # More predictions can be  will be added from a database or session
]

@app.route('/')
def index():
    return render_template('index.html', previous_predictions=previous_predictions)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect user inputs from the HTML form
        year = float(request.form['year'])
        month = float(request.form['month'])
        
        # Map the carrier code (string) to a numeric value using the carrier_mapping
        carrier = carrier_mapping.get(request.form['carrier'], -1)  # Default to -1 if carrier is not found
        
        arr_flights = float(request.form['arr_flights'])
        arr_del15 = float(request.form['arr_del15'])
        carrier_ct = float(request.form['carrier_ct'])
        weather_ct = float(request.form['weather_ct'])
        nas_ct = float(request.form['nas_ct'])
        security_ct = float(request.form['security_ct'])
        late_aircraft_ct = float(request.form['late_aircraft_ct'])
        arr_cancelled = float(request.form['arr_cancelled'])
        arr_diverted = float(request.form['arr_diverted'])

        # Create a DataFrame from user input to make a prediction
        input_data = pd.DataFrame([{
            'year': year,
            'month': month,
            'carrier': carrier,
            'arr_flights': arr_flights,
            'arr_del15': arr_del15,
            'carrier_ct': carrier_ct,
            'weather_ct': weather_ct,
            'nas_ct': nas_ct,
            'security_ct': security_ct,
            'late_aircraft_ct': late_aircraft_ct,
            'arr_cancelled': arr_cancelled,
            'arr_diverted': arr_diverted
        }])

        # Predict the arrival delay using the trained model
        prediction = rf_model.predict(input_data)[0]

        # Store the new prediction in the previous_predictions list 
        previous_predictions.append({
            'year': year,
            'month': month,
            'carrier': request.form['carrier'],
            'predicted_delay': prediction
        })

        # Return the prediction on the HTML page
        return render_template('index.html', prediction_text=f'Predicted Arrival Delay: {prediction:.2f} minutes', previous_predictions=previous_predictions)

if __name__ == '__main__':
    app.run(debug=True)
