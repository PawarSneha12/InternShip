import numpy as np
import pickle as pkl
import streamlit as st



# Load your trained model or define it here
clf = pkl.load(open("./models/xgb_clf.pkl", 'rb'))
reg = pkl.load(open("./models/ensemble_reg.pkl", 'rb'))

#Mappings
status_mapping = {
    'Repaid': 0,
    'Late': 1,
    'Current': 0
}

education_mapping = {
    4.0: 'High School',
    5.0: 'Bachelors',
    3.0: 'Primary',
    1.0: 'Unknown',
    2.0: 'Vocational',
    0.0: 'Unknown',
    -1.0: 'Unknown'
}

marital_mapping = {
    -1.0: 'Unknown',
    3.0: 'Divorced',
    1.0: 'Married',
    2.0: 'Single',
    4.0: 'Widowed',
    5.0: 'Separated',
    0.0: 'Unknown'
}

employment_mapping = {
    -1.0: 'Unknown',
    3.0: 'Retired',
    5.0: 'Self-employed',
    6.0: 'Unemployed',
    4.0: 'Student',
    2.0: 'Employed',
    0.0: 'Unknown'
}

# Create the Streamlit app
def main():
    st.title("Prediction App")

    # Create input fields for the 20 features
    required_features = ['Interest', 'Applied amount', 'Principal payments made',
                         'Interest and penalty payments Made', 'Loan tenure', 'Country FI',
                         'Country ES', 'Marital Status (if Unknown choose \'1\')', 'Employment Status (if Unknown choose \'1\')',
                         'Existing liabilities', 'Total income', 'Credit Score',
                         'Employment status (if Retired choose \'1\')', 'Debt to income ratio', 'Gender (if Unknown choose \'1\')',
                         'MaritalStatus (if Single choose \'1\')', 'Gender (if male choose \'1\')', 'Active late Category (if 180+ choose \'1\')',
                         'Education (if High school choose \'1\')', 'MaritalStatus (if married choose \'1\')']
    
    numeric_inputs = [  'Interest', 
                        'Applied amount', 
                        'Principal payments made',
                        'Interest and penalty payments Made', 
                        'Loan tenure', 
                        'Existing liabilities',
                        'Total income', 
                        'Debt to income ratio'
                    ]
    
    non_numeric_inputs = [  'Country FI', 'Country ES', 'Marital Status (if Unknown choose \'1\')',
                            'Employment Status (if Unknown choose \'1\')', 'Credit Score',
                            'Employment status (if Retired choose \'1\')', 'Gender (if Unknown choose \'1\')', 'MaritalStatus (if Single choose \'1\')',
                            'Gender (if male choose \'1\')', 'Active late Category (if 180+ choose \'1\')', 'Education (if High school choose \'1\')',
                            'MaritalStatus (if married choose \'1\')'
                        ]
    
    feature_inputs = np.zeros(20)
    for i in range(len(numeric_inputs)):
        feature_input = np.float64(st.number_input(f"{numeric_inputs[i]}: ", value=0.0, step=0.5, format="%.6f"))
        feature_inputs[required_features.index(numeric_inputs[i])] = feature_input
        
    for i in range(len(non_numeric_inputs)):
        dropdown_selection = st.selectbox(f"{non_numeric_inputs[i]}: ", [0, 1])
        feature_inputs[required_features.index(non_numeric_inputs[i])] = dropdown_selection

    # Create a submit button
    if st.button("Submit"):
        # Store the user inputs in a dictionary
        user_inputs = {f"feature_{i+1}": value for i, value in enumerate(feature_inputs)}

        # Preprocess the inputs if needed before prediction
        # ...

        # Prepare the input data for prediction (convert dictionary to numpy array)
        input_data = np.array(list(user_inputs.values())).reshape(1, -1)

        # Make the prediction using your model
        reg_prediction = reg.predict(input_data)
        clf_pred = clf.predict(input_data)

        # Display the prediction
        st.write(f"ELA_Mean EMI ROI: {reg_prediction[0]}")
        st.write(f"Status: {clf_pred[0]}")

# Run the app
if __name__ == "__main__":
    main()
