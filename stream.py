import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load the dataset
data = pd.read_csv('Laptop_price.csv')

# Preprocess the data
# Assuming the dataset is clean and ready for modeling
X = data.drop('Price', axis=1)
y = data['Price']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error: {mse}')

# Streamlit app
st.title('Laptop Price Prediction')

# Input fields for user to enter data
st.sidebar.header('Input Features')
brand = st.sidebar.selectbox('Brand', data['Brand'].unique())
processor_speed = st.sidebar.number_input('Processor Speed')
ram_size = st.sidebar.number_input('RAM Size')
storage_capacity = st.sidebar.number_input('Storage Capacity')
screen_size = st.sidebar.number_input('Screen Size')
weight = st.sidebar.number_input('Weight')

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'Brand': [brand],
    'Processor_Speed': [processor_speed],
    'RAM_Size': [ram_size],
    'Storage_Capacity': [storage_capacity],
    'Screen_Size': [screen_size],
    'Weight': [weight]
})

# Convert categorical variables to dummy/indicator variables
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Predict the price
predicted_price = model.predict(input_data)

# Display the predicted price
st.write(f'Predicted Price: {predicted_price[0]}')