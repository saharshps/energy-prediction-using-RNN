import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import pickle

df=pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\ds and ml files\daily_energy_consumption.csv")


print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y' )
df.set_index('Date', inplace=True)

print(df)

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Consumption (kWh)'], label='Energy Consumption')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Daily Energy Consumption Over Time')
plt.legend()
plt.show()

consumption_data = df['Consumption (kWh)']
print(consumption_data)


scaler = MinMaxScaler(feature_range=(0, 1))

scaler_consumption_data = scaler.fit_transform(consumption_data.values.reshape(-1, 1))
print(scaler_consumption_data)

X=[]
y=[]
n_steps=30

for i in range(n_steps, len(scaler_consumption_data)):
    X.append(scaler_consumption_data[i-n_steps:i,0])
    y.append(scaler_consumption_data[i,0])

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

print(consumption_data.resample('M').mean())


model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

(model.fit(X, y, epochs=30, validation_split=0.2))

print(model.summary())

last_30_days = scaler_consumption_data[-n_steps:]
last_30_days_1= last_30_days.reshape(1, n_steps, 1)

predicted_consumption = model.predict(last_30_days)
predicted_consumption_1 = scaler.inverse_transform(predicted_consumption)

predicted_scaled=model.predict(last_30_days)

print(scaler.inverse_transform(predicted_scaled ))

model.save('energy_consumption_model.keras')

pickle.dump(scaler, open('scaler.pkl', 'wb'))