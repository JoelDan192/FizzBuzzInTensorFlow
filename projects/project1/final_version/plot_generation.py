import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
from helpers import *
from proj1_helpers import *
from implementations import *

"""Generate baseline methods plot"""

DATA_TRAIN_PATH = '../train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
y[y==-1]=0
tX,_,_=standardize(tX)

to = int(len(y)*0.5)-1
tX_test, y_test = tX[to:], y[to:]
tX, y = tX[:to], y[:to]



mse_s = []
for pct in [1.0, 0.85, 0.7, 0.55, 0.4]:
    to = int(len(y)*pct)-1
    indices = np.random.permutation(y.shape[0])
    training_idx = indices[:to]
    tXs = tX[training_idx,:]
    ys = y[training_idx]
    w0 = np.zeros(tXs.shape[1])
    _, w_ls = least_squares(ys, tXs)
    _, w_gdls = gradient_descent(ys, tXs, w0, max_iters, gamma)
    _, w_ridge = ridge_regression(ys, tXs, lambda_)
    _, w_lreg_reg = reg_logistic_regression(ys, tXs, lambda_, gamma, max_iters)
    ls_mse = precision(y_test, predict_labels(w_ls,tXs))
    gdls_mse = precision(y_test, predict_labels(w_gdls,tXs))
    ridge_mse = precision(y_test, predict_labels(w_ridge,tXs))
    lreg_reg_loss = precision(y_test, predict_labels_logistic(w_lreg_reg,tXs))
    mse_s.append([ls_mse,gdls_mse,ridge_mse,lreg_reg_loss])

x = np.array([1,2,3,4])

my_xticks = ['ls','gdls','ridge','logreg']
plt.xticks(x, my_xticks)

for results_mse in mse_s:
    plt.plot(x, results_mse, marker='D')
plt.legend(['100%', '85%', '70%','55%','40%'],loc='best')
plt.title('Precision (test) error by varying train set size')
plt.savefig('baseline_methods.jpg', format='jpg')
plt.show()


""" Train/test box plots """
PATH_RAW_DATA = '../train.csv'
PATH_SEL_DATA = '../train_sel.csv'

y_raw, tX_raw, ids = load_csv_data(PATH_RAW_DATA)
y_raw[y_raw==-1]=0
tX_raw,_,_= standardize(tX_raw)

y_sel, tX_sel, ids = load_csv_data(PATH_SEL_DATA)
y_sel[y_sel==-1]=0
tX_sel,_,_= standardize(tX_sel)

from logistic_regression import reg_logistic_regression_auto_plot

max_iters = 1000
lambdas = np.logspace(-5, 2, 10)
kfold = 6

training_errs_raw, test_errs_raw = reg_logistic_regression_auto_plot(y_raw, tX_raw, kfold, max_iters, lambdas)
training_errs_sel, test_errs_sel = reg_logistic_regression_auto_plot(y_sel, tX_sel, kfold, max_iters, lambdas)


fig = plt.figure(1, figsize=(4, 5))

# Create an axes instance
ax = fig.add_subplot(111)
plt.ylabel('Training error on K-fold')

## add patch_artist=True option to ax.boxplot()
## to get fill color
bp = ax.boxplot([training_errs_raw, training_errs_sel], patch_artist=True)
ax.set_xticklabels(['Raw', 'Selected_Poly'])
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

## change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
    # change fill color
    box.set( facecolor = '#1b9e77' )

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
fig.savefig('training_error.jpg', format='jpg',bbox_inches='tight')
