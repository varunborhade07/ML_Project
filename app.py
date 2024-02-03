from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Logistic Regression model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        Test1 = float(request.form['grade1'])
        Test2 = float(request.form['grade2'])
        Test3 = float(request.form['grade3'])
        Attendance = float(request.form['attendance'])
        Interest = request.form['interest'].lower()

        # Convert interest to numerical values if needed
        if Interest == 'low':
            interest_num = 1
        elif Interest == 'medium':
            interest_num = 2
        else:  # 'high'
            interest_num = 3

        # Standardize the input data using the same scaler as in the training phase
        input_data = np.array([[Test1, Test2, Test3, Attendance, interest_num]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Map the numeric prediction to 'Yes' or 'No'
        result = 'Yes' if prediction == 1 else 'No'

        # Define a message based on the result
        if result == 'Yes':
            message = "The student is predicted to be at risk of attrition."
        else:
            message = "The student is predicted to be not at risk of attrition."

        # Render the 'index.html' template with the prediction result and message
        return render_template('index.html', prediction=result, message=message)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('index.html', prediction='Error', message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
