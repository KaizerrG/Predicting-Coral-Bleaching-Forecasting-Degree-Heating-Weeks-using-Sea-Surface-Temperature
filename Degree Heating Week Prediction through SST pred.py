# %% [code] {"execution":{"iopub.status.busy":"2024-10-17T05:56:28.325273Z","iopub.execute_input":"2024-10-17T05:56:28.326010Z","iopub.status.idle":"2024-10-17T05:56:39.487754Z","shell.execute_reply.started":"2024-10-17T05:56:28.325963Z","shell.execute_reply":"2024-10-17T05:56:39.486947Z"},"jupyter":{"outputs_hidden":false}}
from IPython import get_ipython
from keras import backend as K

# %% [code] {"execution":{"iopub.status.busy":"2024-11-01T07:36:45.960053Z","iopub.execute_input":"2024-11-01T07:36:45.960454Z","iopub.status.idle":"2024-11-01T07:36:58.287025Z","shell.execute_reply.started":"2024-11-01T07:36:45.960418Z","shell.execute_reply":"2024-11-01T07:36:58.286016Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd
from pandas import Series
import numpy as np
from numpy import polyfit
from tensorflow.keras.layers import Dense,Activation, Dropout
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from keras.models import Sequential
from keras.utils import plot_model
import time
import math
import keras.callbacks
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from scipy.stats.stats import pearsonr
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,f1_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import itertools
import warnings



# %% [code] {"execution":{"iopub.status.busy":"2024-10-30T07:19:18.349801Z","iopub.execute_input":"2024-10-30T07:19:18.350347Z","iopub.status.idle":"2024-10-30T07:19:22.832671Z","shell.execute_reply.started":"2024-10-30T07:19:18.350304Z","shell.execute_reply":"2024-10-30T07:19:22.831599Z"},"jupyter":{"outputs_hidden":false}}
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example tensor creation
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(x)

# %% [code] {"execution":{"iopub.status.busy":"2024-11-01T07:36:58.288357Z","iopub.execute_input":"2024-11-01T07:36:58.288869Z","iopub.status.idle":"2024-11-01T07:37:04.563375Z","shell.execute_reply.started":"2024-11-01T07:36:58.288836Z","shell.execute_reply":"2024-11-01T07:37:04.562361Z"},"jupyter":{"outputs_hidden":false}}
df=pd.read_csv('/kaggle/input/sst-pred/IndianOcean.csv')
fil_df=df[(df['time']>'2010-01-01') ]

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T16:22:34.259060Z","iopub.execute_input":"2024-10-29T16:22:34.259377Z","iopub.status.idle":"2024-10-29T16:22:44.773056Z","shell.execute_reply.started":"2024-10-29T16:22:34.259345Z","shell.execute_reply":"2024-10-29T16:22:44.772260Z"},"jupyter":{"outputs_hidden":false}}
corals=pd.read_csv('/kaggle/input/corals/deep_sea_corals_6cb7_510e_0a83.csv')
corals['latitude'] = pd.to_numeric(corals['latitude'], errors='coerce')
corals['longitude'] = pd.to_numeric(corals['longitude'], errors='coerce')
fil_corals = corals[
    (corals['latitude'] > -60) & 
    (corals['latitude'] < 31.1859) & 
    (corals['longitude'] > 20.0026) & 
    (corals['longitude'] < 146.8982)
]

# %% [code] {"execution":{"iopub.status.busy":"2024-10-22T06:15:32.428118Z","iopub.execute_input":"2024-10-22T06:15:32.428506Z","iopub.status.idle":"2024-10-22T06:15:32.445732Z","shell.execute_reply.started":"2024-10-22T06:15:32.428464Z","shell.execute_reply":"2024-10-22T06:15:32.444806Z"},"jupyter":{"outputs_hidden":false}}
fil_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-22T06:15:37.040114Z","iopub.execute_input":"2024-10-22T06:15:37.040473Z","iopub.status.idle":"2024-10-22T06:15:37.297811Z","shell.execute_reply.started":"2024-10-22T06:15:37.040437Z","shell.execute_reply":"2024-10-22T06:15:37.296692Z"},"jupyter":{"outputs_hidden":false}}
fil_df.max()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-22T06:15:38.977504Z","iopub.execute_input":"2024-10-22T06:15:38.978198Z","iopub.status.idle":"2024-10-22T06:15:38.983856Z","shell.execute_reply.started":"2024-10-22T06:15:38.978156Z","shell.execute_reply":"2024-10-22T06:15:38.982995Z"},"jupyter":{"outputs_hidden":false}}
fil_df.shape

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2024-10-22T06:15:42.047817Z","iopub.execute_input":"2024-10-22T06:15:42.048494Z","iopub.status.idle":"2024-10-22T06:15:42.245814Z","shell.execute_reply.started":"2024-10-22T06:15:42.048456Z","shell.execute_reply":"2024-10-22T06:15:42.244891Z"},"jupyter":{"outputs_hidden":false}}
fil_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-31T14:04:50.409197Z","iopub.execute_input":"2024-10-31T14:04:50.409525Z","iopub.status.idle":"2024-10-31T14:06:53.197361Z","shell.execute_reply.started":"2024-10-31T14:04:50.409492Z","shell.execute_reply":"2024-10-31T14:06:53.195888Z"},"jupyter":{"outputs_hidden":false}}
# Convert time column to datetime and extract year and month
fil_df['time'] = pd.to_datetime(fil_df['time'])
fil_df['Year'] = fil_df['time'].dt.year
fil_df['Month'] = fil_df['time'].dt.month

# Selecting relevant features (year, month, latitude, longitude)
features = fil_df[['Year', 'Month', 'lat', 'lon']]
target = fil_df['sst']

# Normalize the features using Min-Max Scaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Prepare the dataset for LSTM input (for monthly data)
def create_dataset(X, y, time_step=12):  # Use 12 months as time step for monthly prediction
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        Xs.append(X[i:(i + time_step), :])
        ys.append(y[i + time_step])  # Target is the next month’s SST
    return np.array(Xs), np.array(ys)

# Define time step to capture past 12 months (predicting the next month)
time_step = 12
X, y = create_dataset(features_scaled, target.values, time_step)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model with Dropout and L2 regularization
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), 
               kernel_regularizer=l2(0.05)))  # L2 regularization
model.add(Dropout(0.5))
model.add(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01)))  # L2 regularization
model.add(Dropout(0.5))
model.add(Dense(12, kernel_regularizer=l2(0.05)))  # Dense layer for capturing patterns
model.add(Dense(1))  # Output layer for SST prediction

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1)

# Make predictions for the test set
predictions = model.predict(X_test)

# Display results
print("Model training complete. Predictions made on test set.")

# %% [code] {"execution":{"iopub.status.busy":"2024-11-01T07:37:17.684451Z","iopub.execute_input":"2024-11-01T07:37:17.684915Z"},"jupyter":{"outputs_hidden":false}}
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Convert time column to datetime and extract year and month
fil_df['time'] = pd.to_datetime(fil_df['time'])
fil_df['Year'] = fil_df['time'].dt.year
fil_df['Month'] = fil_df['time'].dt.month

# Selecting the SST target variable
target = fil_df['sst']

# Split the dataset into training and testing sets (80% train, 20% test)
train_size = int(len(target) * 0.8)
train, test = target[:train_size], target[train_size:]

# Check if the data is stationary using the Augmented Dickey-Fuller test
result = adfuller(train)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] > 0.05:
    print("Data is not stationary. Consider differencing or transforming the data.")

# Define ARIMA parameter ranges for grid search
p = range(0,3)
d = range(0,1)
q = range(0,3)
pdq = list(itertools.product(p, d, q))

# Grid search to find the best ARIMA model
best_mse = float("inf")
best_order = None

for param in pdq:
    try:
        model = ARIMA(train, order=param)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))
        mse = mean_squared_error(test, predictions)
        if mse < best_mse:
            best_mse = mse
            best_order = param
    except:
        continue

print(f'Best ARIMA order: {best_order} with MSE: {best_mse}')

# Fit the ARIMA model with the best parameters
model = ARIMA(train, order=best_order)
model_fit = model.fit()
predictions_arima = model_fit.forecast(steps=len(test))

# Evaluate the model performance
mse = mean_squared_error(test, predictions_arima)
print(f'Optimized Mean Squared Error: {mse}')

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test.values, label='Actual SST', color='blue')
plt.plot(test.index, predictions_arima, label='Predicted SST', color='red')
plt.title('Actual vs Predicted SST')
plt.xlabel('Time')
plt.ylabel('Sea Surface Temperature')
plt.legend()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-30T07:19:51.131035Z","iopub.execute_input":"2024-10-30T07:19:51.131930Z","iopub.status.idle":"2024-10-30T07:37:39.871016Z","shell.execute_reply.started":"2024-10-30T07:19:51.131877Z","shell.execute_reply":"2024-10-30T07:37:39.869968Z"},"jupyter":{"outputs_hidden":false}}
fil_df['time'] = pd.to_datetime(fil_df['time'])
fil_df['Year'] = fil_df['time'].dt.year
fil_df['Month'] = fil_df['time'].dt.month

# Selecting relevant features (year, month, latitude, longitude)
features = fil_df[['Year', 'Month', 'lat', 'lon']]
target = fil_df['sst']

# Normalize the features using Min-Max Scaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Prepare the dataset for RNN input (for monthly data)
def create_dataset(X, y, time_step=12):  # Use 12 months as time step for monthly prediction
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        Xs.append(X[i:(i + time_step), :])
        ys.append(y[i + time_step])  # Target is the next month’s SST
    return np.array(Xs), np.array(ys)

# Define time step to capture past 12 months (predicting the next month)
time_step = 24
X, y = create_dataset(features_scaled, target.values, time_step)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the RNN model with Dropout and L2 regularization
model = Sequential()
model.add(SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), 
                    kernel_regularizer=l2(0.05)))  # L2 regularization
model.add(Dropout(0.5))
model.add(SimpleRNN(50, return_sequences=False, kernel_regularizer=l2(0.01)))  # L2 regularization
model.add(Dropout(0.5))
model.add(Dense(12, kernel_regularizer=l2(0.05)))  # Dense layer for capturing patterns
model.add(Dense(1))  # Output layer for SST prediction

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1)

# Make predictions for the test set
predictions_rnn = model.predict(X_test)

# Display results
print("Model training complete. Predictions made on test set.")

# %% [code] {"execution":{"iopub.status.busy":"2024-10-30T07:55:22.309061Z","iopub.execute_input":"2024-10-30T07:55:22.309594Z","iopub.status.idle":"2024-10-30T07:55:22.322119Z","shell.execute_reply.started":"2024-10-30T07:55:22.309555Z","shell.execute_reply":"2024-10-30T07:55:22.321110Z"},"jupyter":{"outputs_hidden":false}}
r2sco = r2_score(y_test, predictions_rnn)
print(r2sco)
rmse=mean_squared_error(y_test,predictions_rnn)
print(rmse)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-18T07:57:40.711160Z","iopub.execute_input":"2024-10-18T07:57:40.711549Z","iopub.status.idle":"2024-10-18T07:57:40.719129Z","shell.execute_reply.started":"2024-10-18T07:57:40.711508Z","shell.execute_reply":"2024-10-18T07:57:40.718233Z"},"jupyter":{"outputs_hidden":false}}
predictions

# %% [code] {"execution":{"iopub.status.busy":"2024-10-17T10:07:35.191781Z","iopub.execute_input":"2024-10-17T10:07:35.192420Z","iopub.status.idle":"2024-10-17T10:07:35.560876Z","shell.execute_reply.started":"2024-10-17T10:07:35.192377Z","shell.execute_reply":"2024-10-17T10:07:35.559624Z"},"jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-17T10:07:38.022562Z","iopub.execute_input":"2024-10-17T10:07:38.023207Z","iopub.status.idle":"2024-10-17T10:07:38.415323Z","shell.execute_reply.started":"2024-10-17T10:07:38.023164Z","shell.execute_reply":"2024-10-17T10:07:38.414411Z"},"jupyter":{"source_hidden":true}}
# Plotting predictions vs actual values
plt.plot(y_test[:7], label='Actual SST', color='blue')  # Plot first 100 actual values
plt.plot(predictions[:7], label='Predicted SST', color='red')  # Plot first 100 predicted values
plt.title('Predicted vs Actual SST LSTM (7 Days)')
plt.ylabel('SST')
plt.xlabel('Sample Index')
plt.legend(loc='upper right')
plt.savefig('l1.png')
plt.show();

# %% [code] {"execution":{"iopub.status.busy":"2024-10-17T15:06:49.202023Z","iopub.execute_input":"2024-10-17T15:06:49.202779Z","iopub.status.idle":"2024-10-17T15:06:49.624399Z","shell.execute_reply.started":"2024-10-17T15:06:49.202737Z","shell.execute_reply":"2024-10-17T15:06:49.623511Z"},"jupyter":{"outputs_hidden":false}}
# Plotting predictions vs actual values
plt.plot(y_test[:30], label='Actual SST', color='blue')  # Plot first 100 actual values
plt.plot(predictions[:30], label='Predicted SST', color='red')  # Plot first 100 predicted values
plt.title('Predicted vs Actual SST LSTM (30 Days)')
plt.ylabel('SST')
plt.xlabel('Sample Index')
plt.legend(loc='upper right')
plt.savefig('l2.png')

# %% [code] {"execution":{"iopub.status.busy":"2024-10-17T07:50:41.127866Z","iopub.execute_input":"2024-10-17T07:50:41.128715Z","iopub.status.idle":"2024-10-17T08:21:38.994266Z","shell.execute_reply.started":"2024-10-17T07:50:41.128673Z","shell.execute_reply":"2024-10-17T08:21:38.993310Z"},"jupyter":{"outputs_hidden":false}}
# Convert time column to datetime and extract year and month
fil_df['time'] = pd.to_datetime(fil_df['time'])
fil_df['Year'] = fil_df['time'].dt.year
fil_df['Month'] = fil_df['time'].dt.month

# Selecting relevant features (year, month, latitude, longitude)
features = fil_df[['Year', 'Month', 'lat', 'lon']]
target = fil_df['sst']

# Normalize the features using Min-Max Scaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Prepare the dataset for BiLSTM input (for monthly data)
def create_dataset(X, y, time_step=12):  # Use 12 months as time step for monthly prediction
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        Xs.append(X[i:(i + time_step), :])
        ys.append(y[i + time_step])  # Target is the next month’s SST
    return np.array(Xs), np.array(ys)

# Define time step to capture past 12 months (predicting the next month)
time_step = 12
X, y = create_dataset(features_scaled, target.values, time_step)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the BiLSTM model with Dropout and L2 regularization
model = Sequential()
# First Bidirectional LSTM layer
model.add(Bidirectional(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), 
                             kernel_regularizer=l2(0.05))))  # L2 regularization
model.add(Dropout(0.75))

# Second Bidirectional LSTM layer 
model.add(Bidirectional(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01))))  # L2 regularization
model.add(Dropout(0.75))

# Dense layer for capturing patterns
model.add(Dense(12, kernel_regularizer=l2(0.05)))  
# Output layer for SST prediction
model.add(Dense(1))  

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1)

# Make predictions for the test set
predi = model.predict(X_test)

# Display results
print("Model training complete. Predictions made on test set.")

# %% [code] {"execution":{"iopub.status.busy":"2024-10-07T06:12:42.249956Z","iopub.execute_input":"2024-10-07T06:12:42.250609Z","iopub.status.idle":"2024-10-07T06:12:42.499235Z","shell.execute_reply.started":"2024-10-07T06:12:42.250570Z","shell.execute_reply":"2024-10-07T06:12:42.497911Z"},"jupyter":{"outputs_hidden":false}}
r2score_bi = r2_score(y_test, predi)
print(r2score_bi)
rmse_bi=mean_squared_error(y_test,predi)
print(rmse_bi)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-26T13:23:33.076780Z","iopub.execute_input":"2024-09-26T13:23:33.077630Z","iopub.status.idle":"2024-09-26T13:23:33.453534Z","shell.execute_reply.started":"2024-09-26T13:23:33.077589Z","shell.execute_reply":"2024-09-26T13:23:33.452610Z"},"jupyter":{"outputs_hidden":false}}
# Plotting predictions vs actual values
plt.plot(y_test[:7], label='Actual SST', color='blue')  # Plot first 100 actual values
plt.plot(predi[:7], label='Predicted SST', color='red')  # Plot first 100 predicted values
plt.title('Predicted vs Actual SST Bi-LSTM(7 Days)')
plt.ylabel('SST')
plt.xlabel('Sample Index')
plt.legend(loc='upper right')
plt.savefig('l3.png')

# %% [code] {"execution":{"iopub.status.busy":"2024-09-26T13:23:34.683985Z","iopub.execute_input":"2024-09-26T13:23:34.684836Z","iopub.status.idle":"2024-09-26T13:23:35.075148Z","shell.execute_reply.started":"2024-09-26T13:23:34.684793Z","shell.execute_reply":"2024-09-26T13:23:35.074152Z"},"jupyter":{"outputs_hidden":false}}
# Plotting predictions vs actual values
plt.plot(y_test[:30], label='Actual SST', color='blue')  # Plot first 100 actual values
plt.plot(predi[:30], label='Predicted SST', color='red')  # Plot first 100 predicted values
plt.title('Predicted vs Actual SST Bi-LSTM(30 Days)')
plt.ylabel('SST')
plt.xlabel('Sample Index')
plt.legend(loc='upper right')
plt.savefig('l4.png')

# %% [code] {"execution":{"iopub.status.busy":"2024-09-25T18:06:00.556074Z","iopub.execute_input":"2024-09-25T18:06:00.556463Z","iopub.status.idle":"2024-09-25T18:06:00.565529Z","shell.execute_reply.started":"2024-09-25T18:06:00.556428Z","shell.execute_reply":"2024-09-25T18:06:00.564398Z"},"jupyter":{"outputs_hidden":false}}
r2sco = r2_score(y_test, predictions)
print(r2sco)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-25T18:06:01.719428Z","iopub.execute_input":"2024-09-25T18:06:01.720136Z","iopub.status.idle":"2024-09-25T18:06:01.727686Z","shell.execute_reply.started":"2024-09-25T18:06:01.720097Z","shell.execute_reply":"2024-09-25T18:06:01.726669Z"},"jupyter":{"outputs_hidden":false}}
rmse=mean_squared_error(y_test,predictions)
print(rmse)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-26T10:27:20.037903Z","iopub.execute_input":"2024-09-26T10:27:20.038971Z","iopub.status.idle":"2024-09-26T10:27:20.376897Z","shell.execute_reply.started":"2024-09-26T10:27:20.038889Z","shell.execute_reply":"2024-09-26T10:27:20.375949Z"},"jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-17T17:27:17.321421Z","iopub.execute_input":"2024-10-17T17:27:17.322516Z","iopub.status.idle":"2024-10-17T17:27:17.336195Z","shell.execute_reply.started":"2024-10-17T17:27:17.322468Z","shell.execute_reply":"2024-10-17T17:27:17.335259Z"},"jupyter":{"outputs_hidden":false}}

columns_to_keep = ['ScientificName', 'latitude','longitude']
fil_corals = fil_corals.loc[:, columns_to_keep]
fil_corals

# %% [code] {"execution":{"iopub.status.busy":"2024-10-18T08:19:19.206856Z","iopub.execute_input":"2024-10-18T08:19:19.207254Z","iopub.status.idle":"2024-10-18T08:20:10.115285Z","shell.execute_reply.started":"2024-10-18T08:19:19.207215Z","shell.execute_reply":"2024-10-18T08:20:10.114390Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Assuming X_test is defined and contains the necessary input sequences
last_input_sequence = X_test[-1]  # Last input sequence from the test set

# List to store future predictions
future_predictions = []

# Number of future months to predict
months_to_predict = 12

# Take latitude, longitude, and start time inputs from the user
try:
    user_lat = float(input("Enter the latitude for prediction (-90 to 90): "))
    user_lon = float(input("Enter the longitude for prediction (-180 to 180): "))
    start_time = input("Enter the start time for prediction (YYYY-MM-DD): ")
    
    # Validate inputs
    if not (-90 <= user_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    if not (-180 <= user_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180.")
    
    # Convert the start_time to a pandas datetime object
    start_time = pd.to_datetime(start_time)

except ValueError as e:
    print(f"Input error: {e}")
    exit()

# Iteratively predict the next SST values for the next 12 months
for month in range(months_to_predict):
    # Reshape the input data to match the model's expected input shape
    reshaped_input = last_input_sequence.reshape((1, last_input_sequence.shape[0], last_input_sequence.shape[1]))
    
    # Predict the monthly SST value (average for the month)
    predicted_monthly_sst = model.predict(reshaped_input)
    
    # Store the predicted monthly SST in the predictions list
    future_predictions.append(predicted_monthly_sst[0, 0])
    
    # Update the input data for the next month
    new_input = np.append(
        reshaped_input[0, 1:, :], 
        [[predicted_monthly_sst[0, 0], reshaped_input[0, -1, 1], user_lat, user_lon]], 
        axis=0
    )
    
    # Update last_input_sequence for the next iteration
    last_input_sequence = new_input

# Create future dates starting from the user-specified start time (first day of each month)
future_dates = pd.date_range(start=start_time, periods=months_to_predict, freq='MS')

# Create a DataFrame for future monthly averages
future_df = pd.DataFrame({'time': future_dates, 'predicted_sst': future_predictions})

# Display the predicted monthly averages for the next 12 months
print("Predicted Monthly Average SST for the next 12 months starting from", start_time.strftime("%Y-%m-%d"))
print(future_df)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-17T17:04:56.326130Z","iopub.execute_input":"2024-10-17T17:04:56.326593Z","iopub.status.idle":"2024-10-17T17:04:56.336470Z","shell.execute_reply.started":"2024-10-17T17:04:56.326554Z","shell.execute_reply":"2024-10-17T17:04:56.335381Z"},"jupyter":{"outputs_hidden":false}}
future_df

# %% [code] {"execution":{"iopub.status.busy":"2024-10-18T07:58:26.495535Z","iopub.execute_input":"2024-10-18T07:58:26.496416Z","iopub.status.idle":"2024-10-18T07:58:26.500586Z","shell.execute_reply.started":"2024-10-18T07:58:26.496370Z","shell.execute_reply":"2024-10-18T07:58:26.499623Z"},"jupyter":{"outputs_hidden":false}}
fut=future_df['predicted_sst']

# %% [code] {"execution":{"iopub.status.busy":"2024-10-18T08:20:19.920241Z","iopub.execute_input":"2024-10-18T08:20:19.920640Z","iopub.status.idle":"2024-10-18T08:20:19.927497Z","shell.execute_reply.started":"2024-10-18T08:20:19.920601Z","shell.execute_reply":"2024-10-18T08:20:19.926652Z"},"jupyter":{"outputs_hidden":false}}
sst=fut.values
sst

# %% [code] {"execution":{"iopub.status.busy":"2024-10-18T08:20:20.647608Z","iopub.execute_input":"2024-10-18T08:20:20.648016Z","iopub.status.idle":"2024-10-18T08:20:20.659930Z","shell.execute_reply.started":"2024-10-18T08:20:20.647977Z","shell.execute_reply":"2024-10-18T08:20:20.658971Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np

def calculate_dhw_with_alerts_monthly(sst, mmm, window=3):
    """
    Calculate Degree Heating Weeks (DHW) with bleaching alerts using monthly SST averages.
    
    Parameters:
    - sst: List or array of monthly sea surface temperatures (SST).
    - mmm: Maximum Monthly Mean temperature.
    - window: Number of months in the rolling period (default is 3 months).
    
    Returns:
    - dhw: Array of Degree Heating Weeks (DHW) values.
    - alerts: List of alert messages when DHW crosses bleaching thresholds.
    """
    # Step 1: Calculate monthly HotSpots (SST - MMM)
    hotspots = np.maximum(0, np.array(sst) - mmm)
    
    # Step 2: Create an empty array to store DHW
    dhw = np.zeros(len(sst))
    alerts = []  # List to store alert messages
    
    # Step 3: Calculate DHW using a rolling window over 3 months
    for i in range(window - 1, len(sst)):
        # Sum the HotSpots over the last 'window' months (e.g., 3 months)
        dhw[i] = np.sum(hotspots[i - window + 1:i + 1]) / 7  # Divide by 7 to convert to DHW
        
        # Step 4: Alert system for bleaching thresholds
        if dhw[i] >= 8:
            alerts.append(f"Month {i + 1}: Excessive bleaching risk (DHW = {dhw[i]:.2f})")
        elif dhw[i] >= 4:
            alerts.append(f"Month {i + 1}: Moderate bleaching risk (DHW = {dhw[i]:.2f})")
        elif dhw[i] >= 2:
            alerts.append(f"Month {i + 1}: Low bleaching risk (DHW = {dhw[i]:.2f})")
        else:  # This covers the case where DHW < 2
            alerts.append(f"Month {i + 1}: Very low bleaching risk (DHW = {dhw[i]:.2f})")
    
    return dhw, alerts

mmm = 22  
dhw_values, bleaching_alerts = calculate_dhw_with_alerts_monthly(sst, mmm)

# Print DHW values and bleaching alerts
print("DHW Values:", dhw_values)
print("\nBleaching Alerts:")
for alert in bleaching_alerts:
    print(alert)

# %% [code] {"jupyter":{"outputs_hidden":false}}
