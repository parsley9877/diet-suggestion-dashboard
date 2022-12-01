from pytorch_tabnet.tab_model import TabNetRegressor

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


import pandas as pd
import numpy as np
np.random.seed(0)


import os
from pathlib import Path

from matplotlib import pyplot as plt

data_path = './data/Food_Supply_kcal_Data.csv'
train = pd.read_csv(data_path)

target = 'Confirmed'

if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train", "valid"], p =[.9, .1], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index

categorical_columns = []
categorical_dims =  {}
print(len(train.columns))
for col in train.columns[train.dtypes == object]:
    print(col, train[col].nunique())
    l_enc = LabelEncoder()
    train[col] = train[col].fillna("VV_likely")
    train[col] = l_enc.fit_transform(train[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

for col in train.columns[train.dtypes == 'float64']:
    train.fillna(train.loc[train_indices, col].mean(), inplace=True)

unused_feat = ['Set']
features = [ col for col in train.columns if col not in unused_feat+[target]]

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# define your embedding sizes : here just a random choice
cat_emb_dim = [5,6,7]

clf = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs, optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=5e-3),)

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices].reshape(-1, 1)

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices].reshape(-1, 1)

max_epochs = 300



clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train),(X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
    max_epochs=max_epochs,
    patience=50,
    batch_size=50, virtual_batch_size=50,
    num_workers=0,
    drop_last=False,
)

loss = clf.history.history['loss']
train_rmsle = clf.history.history['train_rmsle']
train_mae = clf.history.history['train_mae']
train_mse = clf.history.history['train_mse']
train_rmse = clf.history.history['train_rmse']
valid_rmsle = clf.history.history['valid_rmsle']
valid_mae = clf.history.history['valid_mae']
valid_mse = clf.history.history['valid_mse']
valid_rmse = clf.history.history['valid_rmse']



# save tabnet model
saving_path_name = "./tabnet_models/tabnet_model_covid"
saved_filepath = clf.save_model(saving_path_name)

# define new model with basic parameters and load state dict weights
loaded_clf = TabNetRegressor()
loaded_clf.load_model(saved_filepath)
#
loaded_preds = loaded_clf.predict(X_valid)
# print(y_valid)
# print(loaded_preds)
# loaded_test_mse = mean_squared_error(loaded_preds, X_train)
#
# print(f"FINAL TEST SCORE : {loaded_test_mse}")
#
# print(clf.feature_importances_)

fig, ax = plt.subplots()
ax.plot(loss, 'g-')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid()
ax.set_title('TabNet Train Loss vs Epoch - COVID')
fig.savefig('./plots/loss_covid.png')

fig, ax = plt.subplots()
ax.plot(train_rmsle, 'g-')
ax.set_xlabel('Epoch')
ax.set_ylabel('RMSLE')
ax.grid()
ax.set_title('Train Root Mean Squared Logarithmic Error vs Epoch - COVID')
fig.savefig('./plots/train_rmsle_covid.png')

fig, ax = plt.subplots()
ax.plot(valid_rmsle, 'b-')
ax.set_xlabel('Epoch')
ax.set_ylabel('RMSLE')
ax.grid()
ax.set_title('Test Root Mean Squared Logarithmic Error vs Epoch - COVID')
fig.savefig('./plots/valid_rmsle_covid.png')

fig, ax = plt.subplots()
ax.plot(train_mae, 'g-')
ax.set_xlabel('Epoch')
ax.set_ylabel('MAE')
ax.grid()
ax.set_title('Train Mean Absolute Error vs Epoch - COVID')
fig.savefig('./plots/train_mae_covid.png')

fig, ax = plt.subplots()
ax.plot(valid_mae, 'b-')
ax.set_xlabel('Epoch')
ax.set_ylabel('MAE')
ax.grid()
ax.set_title('Test Mean Absolute Error vs Epoch - COVID')
fig.savefig('./plots/valid_mae_covid.png')

fig, ax = plt.subplots()
ax.plot(train_mse, 'g-')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.grid()
ax.set_title('Train Mean Square Error vs Epoch - COVID')
fig.savefig('./plots/train_mse_covid.png')

fig, ax = plt.subplots()
ax.plot(valid_mse, 'b-')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.grid()
ax.set_title('Test Mean Square Error vs Epoch - COVID')
fig.savefig('./plots/valid_mse_covid.png')

fig, ax = plt.subplots()
ax.plot(train_rmse, 'g-')
ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE')
ax.grid()
ax.set_title('Train Root Mean Square Error vs Epoch - COVID')
fig.savefig('./plots/train_rmse_covid.png')

fig, ax = plt.subplots()
ax.plot(valid_rmse, 'b-')
ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE')
ax.grid()
ax.set_title('Test Root Mean Square Error vs Epoch - COVID')
fig.savefig('./valid_rmse_covid.png')
