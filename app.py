import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Processing
# Load the data from CSV file to Pandas DataFrame
titanic_data = pd.read_csv('titanic.csv')

# Handling the missing values
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Encoding categorical columns
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Separating features and target
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']

# Splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# Streamlit App
st.title('Titanic Survival Prediction')

st.write('''
This app predicts if a passenger would have survived the Titanic disaster based on their details.
''')
with st.sidebar:
    st.markdown("## About the Project")
    st.image("titanic.jpeg", use_column_width=True)
    st.markdown("""
    <p style='color: blue; font-weight: bold;'>Developed by Anurag Ghosh</p>
    <p>This application uses a Logistic Regression model to predict the survival of Titanic passengers based on their features such as class, gender, age, number of siblings/spouses aboard, number of parents/children aboard, fare, and port of embarkation.</p>
    <a href="https://github.com/Anurag-ghosh-12/CODSOFT/blob/main/Anurag_Project1_Titanic_Survival_Prediction.ipynb" target="_blank"><button style='background-color: lightblue; padding: 10px; border: none; border-radius: 5px;'>GitHub Repository</button></a>
    """, unsafe_allow_html=True)


# Input fields for user
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.number_input('Age', min_value=0.42, max_value=100.0, step=0.1)
sibsp = st.number_input('Number of Siblings/Spouses aboard', min_value=0, step=1)
parch = st.number_input('Number of Parents/Children aboard', min_value=0, step=1)
fare = st.number_input('Fare', min_value=0.0, step=0.1)
embarked = st.selectbox('Port of Embarkation', ['Southampton', 'Cherbourg', 'Queenstown'])

# Convert inputs to appropriate values
sex = 0 if sex == 'Male' else 1
embarked = 0 if embarked == 'Southampton' else 1 if embarked == 'Cherbourg' else 2

# Prediction
if st.button('Predict Survival'):
    user_input = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(user_input)
    if prediction == 0:
        st.write('Sorry, the passenger would not have survived.')
    else:
        st.write('Congratulations, the passenger would have survived!')

# Display accuracy
st.write(f'Accuracy of the model on training data: {training_data_accuracy*100:.2f}%')
st.write(f'Accuracy of the model on test data: {test_data_accuracy*100:.2f}%')
