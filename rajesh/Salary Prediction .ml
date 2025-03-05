import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Load dataset
A = pd.read_csv(r"/content/salary_prediction_with_expected_salary.csv")

# Exclude the 'Employee Name' column and separate features and target
X = A.iloc[:, 1:7]  # Selecting columns from index 1 to 6
Y = A.iloc[:, 7]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Create a OneHotEncoder object
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit the encoder to the categorical columns and transform them
encoded_data = encoder.fit_transform(X[categorical_cols])

# Create a DataFrame from the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

# Drop the original categorical columns and concatenate the encoded columns
X = X.drop(categorical_cols, axis=1)
X = pd.concat([X, encoded_df], axis=1)

# Standardize the numerical features
numerical_cols = X.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Evaluate the model
print(f"Model score on training data: {model.score(X_train, Y_train):.4f}")
print(f"Model score on testing data: {model.score(X_test, Y_test):.4f}")

# Gradio Prediction Function
def predict_expectedsalary(Age, Experience, No_of_Companies_Worked, Past_Salary, Job, Degree):
    try:
        # Create input data
        input_data = pd.DataFrame([[Age, Experience, No_of_Companies_Worked, Past_Salary, Job, Degree]],
                                  columns=['Age', 'Experience', 'No_of_Companies_Worked', 'Past_Salary', 'Job', 'Degree'])
        
        # Encode categorical features
        categorical_input = input_data[categorical_cols]
        encoded_input_data = encoder.transform(categorical_input)
        encoded_input_df = pd.DataFrame(encoded_input_data, columns=encoder.get_feature_names_out(categorical_cols))

        # Drop original categorical features and concatenate encoded features
        input_data = input_data.drop(categorical_cols, axis=1)
        input_data = pd.concat([input_data, encoded_input_df], axis=1)

        # Scale numerical features
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # Ensure column order matches the training data
        input_data = input_data[X.columns]

        # Make prediction
        model_predict = model.predict(input_data)
        return f"Predicted Expected Salary: {model_predict[0]:,.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio App
iface = gr.Interface(
    fn=predict_expectedsalary,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Experience"),
        gr.Number(label="No of Companies Worked"),
        gr.Number(label="Past Salary"),
        gr.Textbox(label="Job"), 
        gr.Textbox(label="Degree"),  
    ],
    outputs="text",
    title="Expected Salary Prediction"
)

# Launch the application
iface.launch(share=True)
