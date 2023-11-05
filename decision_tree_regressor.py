import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("./consumption_temp.csv")

# Data preprocessing
data['time'] = pd.to_datetime(data['time'])
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour
data['day_of_week'] = data['time'].dt.dayofweek

# One-hot encoding for location (turns categorical data into numerical data)
data = pd.get_dummies(data, columns=['location'], prefix=['Location'])

# Target variable
y = data['consumption']
# Input features
X = data[['year', 'month', 'day', 'hour', 'day_of_week', 'temperature', 'Location_bergen', 'Location_oslo', 'Location_stavanger', 'Location_tromsø', 'Location_trondheim']]

# Split data: Training and testing sets (80% training, 20% testing)
forecast_date = datetime(2023, 9, 19)
# Historical data limitation: 5 days
historical_data_end_date = forecast_date - timedelta(days=5)
train_data = data[data['time'] <= historical_data_end_date]
test_data = data[(data['time'] > historical_data_end_date) & (data['time'] <= forecast_date)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model
aneo_model = DecisionTreeRegressor(random_state=1)
aneo_model.fit(X_train, y_train)

# Predict consumption values for the specified date and time
start_date = datetime(2023, 9, 18)
end_date = forecast_date
date_range = [start_date + timedelta(hours=i) for i in range((end_date - start_date).days * 24)]

predicted_consumption = []

# Input features for the specified date and time
for date in date_range:
    input_features = {
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'hour': [date.hour],
        'day_of_week': [date.weekday()],
        'temperature': [15],  
        # Choose the location by having 1 for the location and 0 for the rest
        'Location_bergen': [1],
        'Location_oslo': [0],
        'Location_stavanger': [0],
        'Location_tromsø': [0],
        'Location_trondheim': [0]
    }

    predicted_consumption_val = aneo_model.predict(pd.DataFrame(input_features))
    predicted_consumption.append(predicted_consumption_val[0])

# Mean Absolute Error 
test_predictions = aneo_model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
print(f'Mean Absolute Error on the test data: {mae:.2f}')

def datetime_to_str(date):
    return date.strftime('%Y/%m/%d')

date_labels = [date.strftime('%Y-%m-%d %H:%M') for date in date_range]

# Predicted consumption values
plt.figure(figsize=(12, 4))
plt.plot(date_labels, predicted_consumption, marker=',', linestyle='-')
plt.title(f'''Bergen: {datetime_to_str(start_date)} - {datetime_to_str(end_date)}''')
plt.ylabel('Consumption (MW)')
plt.grid(False)
plt.xticks(rotation=45)
plt.show()