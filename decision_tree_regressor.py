
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime, timedelta

data = pd.read_csv("./consumption_temp.csv")

y = data['consumption']
aneo_features = ['time', 'location']

data['time'] = pd.to_datetime(data['time'])
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour
data['day_of_week'] = data['time'].dt.dayofweek
data = pd.get_dummies(data, columns=['location'], prefix=['Location'])

X = data[['year', 'month', 'day', 'hour', 'day_of_week', 'temperature', 'Location_bergen', 'Location_oslo', 'Location_stavanger', 'Location_tromsø', 'Location_trondheim']]

aneo_model = DecisionTreeRegressor(random_state=1)
aneo_model.fit(X, y)

# Date range
start_date = datetime(2023, 11, 19)  
end_date = datetime(2023, 12, 23)
date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
predicted_consumption = []

# Consumption prediciton for a specific city and time during the date-range
for date in date_range:
    input_features = {
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'hour': [10],
        'day_of_week': [date.weekday()],
        'temperature': [15],
        'Location_bergen': [1],
        'Location_oslo': [0],
        'Location_stavanger': [0],
        'Location_tromsø': [0],
        'Location_trondheim': [0]
    }

    predicted_consumption_val = aneo_model.predict(pd.DataFrame(input_features))
    predicted_consumption.append(predicted_consumption_val[0])

# Graph
plt.figure(figsize=(12, 4))
plt.plot(date_range, predicted_consumption, marker=',', linestyle='-')
plt.title('Bergen - 10 AM')
plt.ylabel('Consumption (MW)')
plt.grid(False)
plt.xticks(rotation=45)
plt.show()