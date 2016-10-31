
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from proj1_helpers import *
from helpers import *
from implementations import reg_logistic_regression_auto


print("Preparing data")
#Data preprocessing (both for train and test)
df = pd.read_csv('train.csv')
df_t = pd.read_csv('test.csv')

feat_names = [col for col in df.columns if col not in ['Id','Prediction']]

for colname in feat_names:
   df.loc[df[colname]==-999.000,[colname]]=df[colname][df[colname]!=-999.000].median()

df['DER_mass_jet_jet_2'] = df['DER_mass_jet_jet']**2
df['DER_prodeta_jet_jet_2'] = df['DER_prodeta_jet_jet']**2
df['DER_met_phi_centrality_2'] = df['DER_met_phi_centrality']**2
df['DER_lep_eta_centrality_2'] = df['DER_lep_eta_centrality']**2
cols_to_norm = [colname for colname in df.columns if colname.startswith("DER")]

df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: x/x.std())


feat_names_t = [col for col in df_t.columns if col not in ['Id','Prediction']]

for colname in feat_names:
    df_t.loc[df_t[colname]==-999.000,[colname]]=df_t[colname][df_t[colname]!=-999.000].median()
df_t['DER_mass_jet_jet_2'] = df_t['DER_mass_jet_jet']**2
df_t['DER_prodeta_jet_jet_2'] = df_t['DER_prodeta_jet_jet']**2
df_t['DER_met_phi_centrality_2'] = df_t['DER_met_phi_centrality']**2
df_t['DER_lep_eta_centrality_2'] = df_t['DER_lep_eta_centrality']**2
DER_features_t = [colname for colname in df_t.columns if colname.startswith("DER")]

cols_to_norm_t = DER_features_t
df_t[cols_to_norm] = df_t[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
df_t[cols_to_norm] = df_t[cols_to_norm].apply(lambda x: x/x.std())


#Save prepared data in .csv file 
df.to_csv('train_sel.csv',index=False)
df_t.to_csv('test_sel.csv',index=False)


print("Loading prepared data")
#Data loading
DATA_TRAIN_PATH = 'train_sel.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
y[y==-1]=0
ys, tXs = y, tX
tXs,_,_= standardize(tXs)

print("Logistic regression")
#Cumpute error and weights using logistic regression
max_iters = 1000
lambdas = np.logspace(-3, 3, 20)
kfold = 15

w_logreg_auto, logreg_auto_loss = reg_logistic_regression_auto(ys, tXs, kfold, max_iters, lambdas)

print(w_logreg_auto, logreg_auto_loss)

print("Generating predictions")
#Generate predictions and save ouput in csv format for submission
OUTPUT_PATH_LOG = 'submission_log.csv'
DATA_TEST_PATH = 'test_sel.csv' 

_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
tX_test,_,_= standardize(tX_test)


y_pred_log = predict_labels_logistic(w_logreg_auto, tX_test)
create_csv_submission(ids_test, y_pred_log, OUTPUT_PATH_LOG)
