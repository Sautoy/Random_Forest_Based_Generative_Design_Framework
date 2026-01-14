import json
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import sys
sys.path.insert(0, '..')
from utils import create_dir
np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=100000, precision=2)


with open('config_regression.json', 'r') as file:
    config = json.load(file)

raw_data = pd.read_csv(config['data_path'], sep=',')
des_var_names = config['design_variable_names']
inp_var_names = des_var_names + ['Wavevector'] + ['Mode']
feature_names = inp_var_names + ['Frequency']
n_mode = config['num_mode']
n_k = config['num_k']
n_des_vars = len(des_var_names)

data_x = raw_data[inp_var_names].to_numpy()
data_y = raw_data['Frequency'].to_numpy()

folder_name = 'results_rf_regression'

create_dir(folder_name)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, random_state=0)
print('# designs for training:', x_train.shape[0])
print('# designs for testing:', x_test.shape[0])
n_inp = len(inp_var_names)
x_train = x_train.reshape(-1, n_inp)
x_test = x_test.reshape(-1, n_inp)
y_train = y_train.flatten()
y_test = y_test.flatten()

# Get ranges of design variables
des_var_bounds = np.vstack([np.min(x_train[:,:-2], axis=0), np.max(x_train[:,:-2], axis=0)]).T
np.save(f'{folder_name}/des_var_bounds.npy', des_var_bounds)

# Train a random forest
start_time = time.time()
reg = RandomForestRegressor(n_estimators=config['num_estimators'], criterion='squared_error',
                            min_samples_split=2, min_samples_leaf=1, random_state=0)
reg = reg.fit(x_train, y_train)
print('Training time: {:.2f}s'.format(time.time()-start_time))

# class_weight='balanced'

# Evaluate the model on training data
y_pred_train = reg.predict(x_train)
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

# Evaluate the model on test data
y_pred_test = reg.predict(x_test)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

# Print results
print('Training MSE:', mse_train)
print('Training MAE:', mae_train)
print('Training MAPE:', mape_train)
print('Test MSE:', mse_test)
print('Test MAE:', mae_test)
print('Test MAPE:', mape_test)

# Save results to file
lines = [
    'Train MSE: {:.4f}'.format(mse_train),
    'Train MAE: {:.4f}'.format(mae_train),
    'Train MAPE: {:.4f}'.format(mape_train),
    'Test MSE: {:.4f}'.format(mse_test),
    'Test MAE: {:.4f}'.format(mae_test),
    'Test MAPE: {:.4f}'.format(mape_test)
]

with open(f'{folder_name}/forest_regression_acc.txt', 'w') as f:
    f.write('\n'.join(lines))

# Save the model
dump(reg, f'{folder_name}/forest.joblib')