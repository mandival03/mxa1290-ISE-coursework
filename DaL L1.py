import pandas as pd
import numpy as np
import os
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler  # 🔄 NEW: MinMaxScaler for normalization

systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
train_frac = 0.7
num_repeats = 3
random_seed = 1
dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")  # You should place the dataset folders here


results = []


for system in systems:
    system_path = os.path.join(dataset_root, system)
    if not os.path.isdir(system_path):
        continue

    for filename in os.listdir(system_path):
        if not filename.endswith(".csv"):
            continue

        print(f"Processing: {system}/{filename}")

        filepath = os.path.join(system_path, filename)
        data = pd.read_csv(filepath)

        
        metrics_baseline = {'MAPE': [], 'MAE': [], 'RMSE': [], 'TIME': []}
        metrics_dal_l1 = {'MAPE': [], 'MAE': [], 'RMSE': [], 'TIME': []}

        for repeat in range(num_repeats):
            
            train_data = data.sample(frac=train_frac, random_state=random_seed * repeat)
            test_data = data.drop(train_data.index)

            X_train = train_data.iloc[:, :-1]
            y_train = train_data.iloc[:, -1]
            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            
            start_baseline = time.time()

            baseline_model = LinearRegression()
            baseline_model.fit(X_train, y_train)
            preds_baseline = baseline_model.predict(X_test)

            end_baseline = time.time()

            metrics_baseline['MAPE'].append(mean_absolute_percentage_error(y_test, preds_baseline))
            metrics_baseline['MAE'].append(mean_absolute_error(y_test, preds_baseline))
            metrics_baseline['RMSE'].append(np.sqrt(mean_squared_error(y_test, preds_baseline)))
            metrics_baseline['TIME'].append(end_baseline - start_baseline)

            
            start_dal = time.time()

            
            tree = DecisionTreeRegressor(max_leaf_nodes=3, random_state=random_seed)
            tree.fit(X_train, y_train)
            leaf_ids_train = tree.apply(X_train)
            leaf_ids_test = tree.apply(X_test)

            
            local_models = {}
            scalers = {}

            for leaf in np.unique(leaf_ids_train):
                idx = (leaf_ids_train == leaf)
                X_leaf = X_train[idx]
                y_leaf = y_train[idx]

                scaler = MinMaxScaler()
                X_leaf_scaled = scaler.fit_transform(X_leaf)

                l1_model = Lasso(alpha=0.001, max_iter=10000)
                l1_model.fit(X_leaf_scaled, y_leaf)

                local_models[leaf] = l1_model
                scalers[leaf] = scaler  

            
            preds_dal = []
            for i, leaf in enumerate(leaf_ids_test):
                if leaf in local_models:
                    X_input = X_test.iloc[[i]]
                    X_input_scaled = scalers[leaf].transform(X_input)
                    pred = local_models[leaf].predict(X_input_scaled)[0]
                else:
                    pred = np.mean(y_train)
                preds_dal.append(pred)

            end_dal = time.time()

            metrics_dal_l1['MAPE'].append(mean_absolute_percentage_error(y_test, preds_dal))
            metrics_dal_l1['MAE'].append(mean_absolute_error(y_test, preds_dal))
            metrics_dal_l1['RMSE'].append(np.sqrt(mean_squared_error(y_test, preds_dal)))
            metrics_dal_l1['TIME'].append(end_dal - start_dal)

        
        results.append({
            "system": system,
            "dataset": filename,
            "baseline_MAE": np.mean(metrics_baseline['MAE']),
            "baseline_MAPE": np.mean(metrics_baseline['MAPE']),
            "baseline_RMSE": np.mean(metrics_baseline['RMSE']),
            "baseline_Time(s)": np.mean(metrics_baseline['TIME']),
            "dal_MAE": np.mean(metrics_dal_l1['MAE']),
            "dal_MAPE": np.mean(metrics_dal_l1['MAPE']),
            "dal_RMSE": np.mean(metrics_dal_l1['RMSE']),
            "dal_Time(s)": np.mean(metrics_dal_l1['TIME']),
        })


df = pd.DataFrame(results)
pd.set_option('display.max_rows', None)
print(df)

