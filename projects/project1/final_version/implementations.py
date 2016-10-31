import numpy as np
import costs

from helpers import batch_iter

"""backtracking line search
Unconstrained Optimization, Chapter- Convex Optimization, Boyd and Vandenberghe"""

def backtracking_line_search(alpha, beta, x, f, gf, deltax):
    if (alpha<0 or alpha>0.5) or (beta<0 or beta>1):
        raise ValueError
    slope = np.dot(gf.T,deltax)
    intercept = f(x)
    t=1.0
    while f(x+t*deltax)>=(intercept+alpha*t*slope):
        t=beta*t
    return t


"""cross-validation toolbox"""

def build_k_indices(num_row, k_fold, seed):
    """build k indices for k-fold."""
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_step(y, x, k_indices, k, lambda_, regression_method, regression_loss):
    """@input: regression_method(y, tx, lambda_)
               regression_loss(y, tx, w)."""
    test_idx = k_indices[k,:]
    row_idx = list(range(k_indices.shape[0]))
    train_idx = k_indices[row_idx[:k]+row_idx[k+1:],:].flatten()
    y_train, x_train = y[train_idx], x[train_idx, :]
    y_test, x_test = y[test_idx], x[test_idx, :]
    w,loss_tr = regression_method(y_train, x_train, lambda_)
    loss_te = regression_loss(y_test, x_test, w, lambda_)
    return (loss_tr, loss_te)

def cross_validation(y, x, k_fold, lambdas, regression_method, regression_loss):
    """returns one (rmse_tr, rmse_te) e.g. rmse training/test error per lambda, obtained
    via avaraging in a k-fold cross validation"""
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y.shape[0], k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    #We use the mean scaled by the variance and sample size as performance metric on the test error across folds
    for lamb in lambdas:
        e_tr, e_te = zip(*[cross_validation_step(y, x, k_indices, k, lamb, regression_method, regression_loss) for k in range(k_fold)])
        e_tr, e_te = np.array(e_tr), np.array(e_te)
        e_tr = e_tr.mean()*(k_fold*e_tr.std())
        e_te = e_te.mean()*(k_fold*e_te.std())
        rmse_tr.append(e_tr)
        rmse_te.append(e_te)
    return zip(lambdas, rmse_tr, rmse_te)


def cross_validation_plot(y, x, k_fold, lambdas, regression_method, regression_loss):
    """returns one (rmse_tr, rmse_te) e.g. rmse training/test error per lambda, obtained
    via avaraging in a k-fold cross validation"""
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y.shape[0], k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    #We use the mean scaled by the variance and sample size as performance metric on the test error across folds
    for lamb in lambdas:
        e_tr, e_te = zip(*[cross_validation_step(y, x, k_indices, k, lamb, regression_method, regression_loss) for k in range(k_fold)])
        e_tr, e_te = np.array(e_tr), np.array(e_te)
        rmse_tr = rmse_tr + [e for e in e_tr]
        rmse_te = rmse_te + [e for e in e_te]
    return (rmse_tr, rmse_te)

"""
Split data.
"""
def split_data(y, x, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    if ratio<0 or ratio>1:
        raise ValueError("Train ratio must be betwen 0 and 1")
    np.random.seed(seed)
    n = y.shape[0]
    indices = np.random.permutation(n)
    train_lim = int(n*ratio)
    train_idx, test_idx = indices[:train_lim], indices[train_lim:]
    y_train, y_test = y[train_idx], y[test_idx]
    x_train, x_test = x[train_idx,:], x[test_idx,:]
    return (y_train, x_train, y_test, x_test)

"""functions in the least_squares regression family"""
   
def least_squares(y, tx):
    """calculate the least squares solution using normal equations."""
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))
    return (w, costs.compute_loss(y, tx, w))

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.dot(tx,w)
    n = e.shape[0]
    return -(1/n)*np.dot(tx.T,e)
        
def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm with interface as asked in
    the project description"""
    loss = None
    w = initial_w
    threshold = 1e-8
    losses = []
    
    for n_iter in range(max_iters):
        g = compute_gradient(y,tx,w)
        if np.sum(np.isnan(g))>0:
            print(w)
            print(g)
            return
        loss = costs.compute_loss(y, tx, w)
        w = w - gamma*g
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return (w,losses[-1])


def least_squares_GD_auto(y, tx, max_iters, initial_w): 
    """Gradient descent algorithm with backtracking line-search"""
    loss = None
    w = initial_w
    threshold = 1e-8
    
    def loss_w(w):
        return costs.compute_loss(y, tx, w)   
    for n_iter in range(max_iters):
        g = compute_gradient(y,tx,w)
        loss = costs.compute_loss(y, tx, w)
        if np.dot(g.T,g)<threshold:
            break      
        t = backtracking_line_search(0.4, 0.8, w, loss_w, g,-g)
        w = w - t*g       
    return (w,loss)

def least_squares_SGD_simple(y, tx, max_iters, batch_size, initial_w, gamma): 
    """Stochastic gradient descent"""
    loss = None
    loss_prev = None
    w = initial_w
    threshold = 1e-8
    
    def loss_w(w):
        return costs.compute_loss(y, tx, w)   
    for n_iter in range(max_iters):
        g = compute_stoch_gradient(y,tx,w,batch_size)
        loss_prev = loss
        loss = costs.compute_loss(y, tx, w)
        delta_w = -g
        if loss_prev and np.abs(loss-loss_prev) < threshold:
            return (w, loss)
        w = w + gamma*delta_w       
    return (w,loss)


def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient for batch data."""
    yn, xn = next(batch_iter(y,tx,batch_size))
    residuals_batch = np.array(list(map(lambda tup: -(tup[0]-np.dot(tup[1],w.T)), zip(yn,xn))))
    return (1/batch_size)*np.dot(residuals_batch, xn)


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm with interface as asked in
    the project description"""
    # Define parameters to store w and loss
    loss = None
    w = initial_w
    threshold = 1e-8
    losses = []
    for n_iter in range(max_epochs):
        g = compute_stoch_gradient(y,tx,w,batch_size)
        loss = costs.compute_loss(y, tx, w)
        w = w - gamma*g
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return (w,losses[-1])


def least_squares_GD(y, tx, initial_w, max_iters, gamma):    
    return gradient_descent(y, tx, initial_w, max_iters, gamma)
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):    
    batch_size = 1
    return stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma)

"""ridge regression using normal equations"""

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    n = y.shape[0]
    w = np.linalg.solve(np.dot(tx.T,tx)+ lambda_*np.eye(tx.shape[1]), np.dot(tx.T,y))
    return (w, costs.compute_loss(y, tx, w))

def ridge_regression_auto(y, tx, kfold, lambdas):
    """ridge regression with kfold cross-validation"""
    
    #this loss is used with data split in cross_validation. keep parameters generic
    def ridge_loss(ys, xs, ws, lambda_):
        return costs.compute_loss(ys, xs, ws) + lambda_*np.dot(ws.T,ws)
        
    #choose lambda according to avg test error across folds
    best_lbda = max(cross_validation(y, tx, kfold, lambdas, ridge_regression, ridge_loss)
                    ,key=lambda tup: tup[2])[0]
    train_ratio = 1-(1/float(kfold))
    y_train, x_train, y_test, x_test = split_data(y, tx, train_ratio)
    w, train_mse = ridge_regression(y_train, x_train, best_lbda)
    #Model selection (kfold) uses cost with lbda, we return un-regularized cost.
    test_mse = costs.compute_loss(y_test, x_test, w)
    return (w,test_mse)


"""functions in the logistic regression family"""

def sigmoid(x):
    "Numerically-stable sigmoid function."
    return np.exp(-np.logaddexp(0, -x))


def calculate_loss(y, tx, w):
    sig = sigmoid(np.dot(tx,w))
    class_one_log_probs = -np.sum(y*np.log(sig))
    class_two_log_probs = -np.sum((1-y)*np.log(1-sig))
    return class_one_log_probs+class_two_log_probs

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T, sigmoid(np.dot(tx,w))-y)

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    Sig_txw = sigmoid(np.dot(tx,w))
    S = (Sig_txw*(1-Sig_txw)).flatten()
    return np.dot((tx.T)*S,tx)

def logistic_regression_tuple(y, tx, w):
    """return the loss, gradient, and hessian."""
    return (calculate_loss(y, tx, w), calculate_gradient(y, tx, w), calculate_hessian(y, tx, w))

def penalized_logistic_regression_tuple(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w) + lambda_*np.dot(w.T,w)
    g = calculate_gradient(y, tx, w) + 2*lambda_*w
    h = calculate_hessian(y, tx, w) + 2*lambda_*np.eye(w.shape[0])
    return (loss, g, h)

def newton_step(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, g, h = logistic_regression_tuple(y, tx, w)
    w = w - gamma*np.dot(np.linalg.inv(h),g)
    return loss, w

def penalized_newton_step(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, g, h = penalized_logistic_regression_tuple(y, tx, w, lambda_)
    delta_w = -np.dot(np.linalg.inv(h),g)
    w0=w
    return (loss, w + gamma*delta_w, w0, 0.5*np.dot(g.T,-delta_w))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    debug_mode = False
    threshold = 1e-8
    loss = None
    loss_prev = None  
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = newton_step(y, tx, w, gamma)
        # log info
        if debug_mode and iter % 500 == 0:
            print("Current iteration={i}, the loss={l}, gn={gn}".format(i=iter, l=loss, gn=np.dot(g.T,g)))
            print("weights={ws}".format(ws=w))
        # converge criteria
        if  loss_prev and np.abs(loss - loss_prev) < threshold:
            break
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    debug_mode = False
    threshold = 1e-10
    loss_prev = 0
    loss = None
    
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w, w0, gr_norm = penalized_newton_step(y, tx, w, gamma, lambda_)
        # log info
        if debug_mode and iter % 500 == 0:
            print("Current iteration={i}, the loss={l}, gn={gn}".format(i=iter, l=loss, gn=np.dot(g.T,g)))
            print("weights={ws}".format(ws=w))
        # converge criteria
        if gr_norm<threshold:
            w=w0
            break
        if loss<0:
            raise ValueError("Loss functions should be non-negative")
        if np.abs(loss - loss_prev) < threshold:
            w=w0
            break
        loss_prev = loss
    #returns non-reg loss. to compare to non-reg methods.
    return (w,calculate_loss(y, tx, w))

def logistic_auto(y, tx, lambda_, max_iters, initial_w):
    """Logistic regression with Newton's method and backtracking line-search."""
    loss = None
    w = initial_w
    threshold = 1e-10
    loss_prev = 0
    loss = None
    
    def loss_w(w):
        return calculate_loss(y, tx, w) + lambda_*np.dot(w.T,w)   
    for n_iter in range(max_iters):
        g = calculate_gradient(y,tx,w) + 2*lambda_*w
        loss = calculate_loss(y, tx, w) + lambda_*np.dot(w.T,w)
        h = calculate_hessian(y, tx, w) + 2*lambda_*np.eye(w.shape[0])
        delta_w = -np.dot(np.linalg.inv(h),g)
        #check grad(f)~=0 in the quadratic norm of hessian.
        if 0.5*np.dot(g.T,-delta_w) < threshold or (loss_prev and np.abs(loss - loss_prev) < threshold):
            break
        t = backtracking_line_search(0.45, 0.9, w, loss_w, g,delta_w)
        w = w + t*delta_w
        loss_prev = loss
    #we use it for cross-validation so we keep penalized loss
    return ( w, loss)

def reg_logistic_regression_auto(y, tx, kfold, max_iters, lambdas):
    
    initial_w = np.zeros(tx.shape[1])

    def curried_logistic(ys, xs, lambda_):
        return logistic_auto(ys, xs, lambda_, max_iters, initial_w)
    def logistic_reg_loss(ys, xs, ws, lambda_):
        return calculate_loss(ys, xs, ws) + lambda_*np.dot(ws.T,ws)
    
    best_lbda = min(cross_validation(y, tx, kfold, lambdas, curried_logistic, logistic_reg_loss),
        key = lambda tup: tup[2])[0]
  
    train_ratio = 1-(1/float(kfold))
    y_train, x_train, y_test, x_test = split_data(y, tx, train_ratio)
    w, train_loss = curried_logistic(y_train, x_train, best_lbda)
    
    #Model selection (kfold) uses cost with lbda, we return un-regularized cost.
    test_loss = calculate_loss(y_test, x_test, w)
    return (w, test_loss)


def reg_logistic_regression_auto_plot(y, tx, kfold, max_iters, lambdas):
    initial_w = np.zeros(tx.shape[1])
    def curried_logistic(ys, xs, lambda_):
        return logistic_auto(ys, xs, lambda_, max_iters, initial_w)
    def logistic_reg_loss(ys, xs, ws, lambda_):
        return calculate_loss(ys, xs, ws) + lambda_*np.dot(ws.T,ws)
    
    return cross_validation_plot(y, tx, kfold, lambdas, curried_logistic, logistic_reg_loss)
  
    
    
    
