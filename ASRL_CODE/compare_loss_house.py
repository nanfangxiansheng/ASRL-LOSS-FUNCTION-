import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, recall_score
from sklearn.ensemble import GradientBoostingRegressor as XGBoost
from sklearn.preprocessing import StandardScaler
from test import ASRLXGBoost  #  ASRLXGBoost
import time
import matplotlib.pyplot as plt

#load the data
data = fetch_california_housing()
X, y = data.data, data.target


#divide the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Repair the standardization of y_train
y_mean, y_std = np.mean(y_train), np.std(y_train)
y_train = (y_train - y_mean) / y_std

# USE ASRLXGBoost
timeAS_start = time.time()
asrl_model = ASRLXGBoost(n_estimators=100, learning_rate=0.1, maxdepth=7)
asrl_model.fit(X_train, y_train)
asrl_predictions = asrl_model.predict(X_test) * y_std + y_mean
timeAS_stop = time.time()
print("\nASRLXGBoost finished")
# USE MSEXGBoost
timeMSE_start = time.time()
mse_model = XGBoost(n_estimators=100, learning_rate=0.1, loss='squared_error', random_state=42)
mse_model.fit(X_train, y_train)
mse_predictions = mse_model.predict(X_test) * y_std + y_mean
timeMSE_stop = time.time()
print("\nMSEXGBoost finished")
#USE MAE loss
timeMAE_start = time.time()
mae_model = XGBoost(n_estimators=100, learning_rate=0.1, loss='absolute_error', random_state=42)
mae_model.fit(X_train, y_train)
mae_predictions = mae_model.predict(X_test) * y_std + y_mean
time_MAE_stop = time.time()
print("\nMAEXGBoost finished")
#USE huber loss
timehuber_start = time.time()
huber_model = XGBoost(n_estimators=100, learning_rate=0.1, loss='huber', random_state=42)
huber_model.fit(X_train, y_train)
huber_predictions = huber_model.predict(X_test) * y_std + y_mean
timehuber_stop = time.time()
print("\nHuberXGBoost finished")
# calculate the performance metrics
asrl_mse = mean_squared_error(y_test, asrl_predictions)#asrl->MSEtest
mse_mse = mean_squared_error(y_test, mse_predictions)#mse->MSEtest  
mae_mse = mean_squared_error(y_test, mae_predictions)#mae->MSEtest
huber_mse = mean_squared_error(y_test, huber_predictions)#huber->MSEtest
asrl_mae = mean_absolute_error(y_test, asrl_predictions)    #asrl->MAEtest
mse_mae = mean_absolute_error(y_test, mse_predictions)#mse->MAEtest
mae_mae = mean_absolute_error(y_test, mae_predictions)#mae->MAEtest
huber_mae = mean_absolute_error(y_test, huber_predictions)#huber->MAEtest
asrl_r2 = r2_score(y_test, asrl_predictions)#asrl->R2test
mse_r2 = r2_score(y_test, mse_predictions)#mse->R2test
mae_r2 = r2_score(y_test, mae_predictions)#mae->R2test
huber_r2 = r2_score(y_test, huber_predictions)#huber->R2test

#turn the continuous value into binary classification label
threshold = np.median(y_test)
y_test_binary = (y_test >= threshold).astype(int)
asrl_predictions_binary = (asrl_predictions >= threshold).astype(int)
mse_predictions_binary = (mse_predictions >= threshold).astype(int)
mae_predictions_binary = (mae_predictions >= threshold).astype(int)
huber_predictions_binary = (huber_predictions >= threshold).astype(int)
# calculate the recall
asrl_recall = recall_score(y_test_binary, asrl_predictions_binary)
mse_recall = recall_score(y_test_binary, mse_predictions_binary)
mar_recall=recall_score(y_test_binary,mae_predictions_binary)
huber_recall=recall_score(y_test_binary,huber_predictions_binary)
#output data predict verify
print(y_test[0:10])
print(asrl_predictions[0:10])
print(mse_predictions[0:10])
# output the performance metrics
print("\nPerformance Metrics:")
print("\nMSE:")
print(f"ASRLXGBoost MSE: {asrl_mse:.4f}")
print(f"MSEXGBoost MSE: {mse_mse:.4f}")
print(f"MAEXGBoost MSE: {mae_mse:.4f}")
print(f"HuberXGBoost MSE: {huber_mse:.4f}")
print("\nMAE:")
print(f"ASRLXGBoost MAE: {asrl_mae:.4f}")
print(f"MSEXGBoost MAE: {mse_mae:.4f}")
print(f"MAEXGBoost MAE: {mae_mae:.4f}")
print(f"HuberXGBoost MAE: {huber_mae:.4f}")
print("\nR²:")
print(f"ASRLXGBoost R²: {asrl_r2:.4f}")
print(f"MSEXGBoost R²: {mse_r2:.4f}")   
print(f"MAEXGBoost R²: {mae_r2:.4f}")
print(f"HuberXGBoost R²: {huber_r2:.4f}")
print("\nRecall:")
print(f"ASRLXGBoost Recall: {asrl_recall:.4f}")
print(f"MSEXGBoost Recall: {mse_recall:.4f}")
print(f"MAEXGBoost Recall: {mar_recall:.4f}")
print(f"HuberXGBoost Recall: {huber_recall:.4f}")
# print the time consumption
print("\nTime Consumption:")
print(f"ASRLXGBoost Training Time: {timeAS_stop - timeAS_start:.2f} seconds")
print(f"MSEXGBoost Training Time: {timeMSE_stop - timeMSE_start:.2f} seconds")
print(f"MAEXGBoost Training Time: {time_MAE_stop - timeMAE_start:.2f} seconds")
print(f"HuberXGBoost Training Time: {timehuber_stop - timehuber_start:.2f} seconds")
# Visualize ASRLXGBoost predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='orange', label='Actual Values (y_test)', alpha=0.7)
plt.scatter(range(len(asrl_predictions)), asrl_predictions, color='red', label='Predicted Values (asrl_predictions)', alpha=0.7)
plt.title('ASRLXGBoost Predictions vs Actual Values', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('House Price', fontsize=12)
plt.legend(fontsize=12)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#note:
#this code compare the performace of ASRL with other four loss function 
#based on the dataset of california housing price
#note that the output value varies everytime
