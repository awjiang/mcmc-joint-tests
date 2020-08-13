import multiprocessing
import numpy as onp
# import shogun as sg
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import ks_2samp, chisquare, rankdata
from scipy.special import perm
from arch.covariance.kernel import Bartlett
import os
import pickle
from time import perf_counter

''' 
Split `num_iter` iterations into `nproc` chunks for multithreading
'''
def splitIter(num_iter, nproc):
    arr_iter = (onp.zeros(nproc) + num_iter // nproc).astype('int')
    for i in range(num_iter % nproc):
        arr_iter[i] += 1
    return arr_iter

#######################################################################
############################# Kernels #################################
#######################################################################

class kernel(object):
    def __init__(self, X, Y):
        self._X = X
        self._Y = Y
        pass

    @property
    def params(self):
        pass

    def set_params(self, params):
        pass
    
    def eval(self):
        pass

    def f_kernel(self):
        pass

''' 
Generate RBF kernel matrix with width `tau`. 
If `tau` is None, will use the median heuristic.
Each row corresponds to an observation (row) from X; each column corresponds to an observation (row) from Y.
'''
class rbf_kernel(kernel):
    def __init__(self, X, Y, tau=None):
        assert X.shape[1] == Y.shape[1]
        assert len(X.shape) == 2 and len(X.shape) == len(Y.shape)
        if tau is not None:
            assert isinstance(tau, int) or isinstance(tau, float)
        self._X = X
        self._Y = Y
        self._tau = tau
        pass

    @property
    def params(self):
        return self._tau

    def set_params(self, params):
        self._tau = params
        pass

    def learn(self, method='median_heuristic', eval=False):
        assert method in ['median_heuristic']
        n_X, p = self._X.shape
        n_Y = self._Y.shape[0]
        if method == 'median_heuristic':
            norm2 = ((self._X.reshape(n_X, 1, p) - self._Y.reshape(1, n_Y, p))**2).sum(2)
            norm2_sorted = onp.sort(norm2.reshape(n_X*n_Y, 1).flatten())
            if (n_X*n_Y) % 2 == 0:
                tau = (norm2_sorted[int(n_X*n_Y/2)-1] + norm2_sorted[int(n_X*n_Y/2)])/2
            else:
                tau = norm2_sorted[int(n_X*n_Y/2)]
        else:
            raise ValueError
        self._tau = tau

        if eval == True:
            return onp.exp(-norm2/self._tau)
        else:
            pass

    def eval(self):
        if self._tau is None:
            K = self.learn(eval=True)
        else:
            n_X, p = self._X.shape
            n_Y = self._Y.shape[0]
            K = onp.exp(-(((self._X.reshape(n_X, 1, p) - self._Y.reshape(1, n_Y, p))**2).sum(2))/self._tau)
        return K

    def f_kernel(self, x, y, tau=None):
        if tau is None:
            tau = self._tau
        assert len(x.shape) == len(y.shape) and len(x.shape) == 1
        return onp.exp(-((x-y)**2).sum()/tau)

class sum_uni_rbf_kernel(rbf_kernel):
    def __init__(self, X, Y, tau=None):
        assert X.shape[1] == Y.shape[1]
        assert len(X.shape) == 2 and len(X.shape) == len(Y.shape)
        if tau is not None:
            if isinstance(tau, onp.ndarray):
                tau = tau.reshape(1, 1, X.shape[1])
            else:
                assert isinstance(tau, int) or isinstance(tau, float)
        self._X = X
        self._Y = Y
        self._tau = tau
        pass

    def learn(self, method='median_heuristic', eval=False):
        n_X, p = self._X.shape
        n_Y = self._Y.shape[0]
        if method == 'median_heuristic':
            norm2 = ((self._X.reshape(n_X, 1, p) - self._Y.reshape(1, n_Y, p))**2)
            norm2_sorted = onp.sort(norm2.reshape(n_X*n_Y,p,order='F'), axis=0)
            if (n_X*n_Y) % 2 == 0:
                tau = (norm2_sorted[int(n_X*n_Y/2)-1, :] + norm2_sorted[int(n_X*n_Y/2), :])/2
            else:
                tau = norm2_sorted[int(n_X*n_Y/2), :]
        tau = tau.reshape(1,1,p)
        self._tau = tau
        if eval == True:
            return onp.exp(-norm2/self._tau).sum(2)
        else:
            pass        

    def eval(self):
        if self._tau is None:
            K = self.learn(eval=True)
        else:
            n_X, p = self._X.shape
            n_Y = self._Y.shape[0]
            K = onp.exp(-((self._X.reshape(n_X, 1, p) - self._Y.reshape(1, n_Y, p))**2)/self._tau).sum(2)
        return K

    def f_kernel(self, x, y, tau=None):
        if tau is None:
            tau = self._tau
        assert len(x.shape) == len(y.shape) and len(x.shape) == 1
        return onp.exp(-((x-y)**2)/tau).sum()
# ''' 
# Generate RBF kernel matrix with width `tau`. 
# If `tau` is None, will use the median heuristic.
# Each row corresponds to an observation (row) from X; each column corresponds to an observation (row) from Y.
# '''
# def rbf_kernel(X, Y, tau=None):
#     assert X.shape[1] == Y.shape[1] and len(X.shape) == len(Y.shape) and len(X.shape) == 2
#     n_X, p = X.shape
#     n_Y = Y.shape[0]

#     norm2 = ((X.reshape(n_X, 1, p) - Y.reshape(1, n_Y, p))**2).sum(2)
#     if tau is None:
#         norm2_sorted = onp.sort(norm2.reshape(n_X*n_Y, 1).flatten())
#         if (n_X*n_Y) % 2 == 0:
#             tau = (norm2_sorted[int(n_X*n_Y/2)-1] + norm2_sorted[int(n_X*n_Y/2)])/2
#         else:
#             tau = norm2_sorted[int(n_X*n_Y/2)]

#     K = onp.exp(-norm2/tau)
#     return K

# ''' 
# Generate sum of univariate RBF kernel matrices with width `tau`.
# If `tau` is None, will use the median heuristic.
# Each row corresponds to an observation (row) from X; each column corresponds to an observation (row) from Y.
# '''
# def sum_rbf_kernel(X, Y, tau=None):
#     assert X.shape[1] == Y.shape[1] and len(X.shape) == len(Y.shape) and len(X.shape) == 2
#     n_X, p = X.shape
#     n_Y = Y.shape[0]

#     norm2 = ((X.reshape(n_X, 1, p) - Y.reshape(1, n_Y, p))**2)
#     if tau is None:
#         norm2_sorted = onp.sort(norm2.reshape(n_X*n_Y,p,order='F'), axis=0)
#         if (n_X*n_Y) % 2 == 0:
#             tau = (norm2_sorted[int(n_X*n_Y/2)-1, :] + norm2_sorted[int(n_X*n_Y/2), :])/2
#         else:
#             tau = norm2_sorted[int(n_X*n_Y/2), :]

#     tau = tau.reshape(1,1,p)
#     K = onp.exp(-norm2/tau).sum(2)
#     return K

#######################################################################
############################# Geweke test #############################
#######################################################################

# '''
# Calculate the squared standard error of the estimate of E[`g`] (Geweke 1999, 3.7-8).
# '''
# def geweke_se2(g, L=0, center=True):
#     L = int(L)
#     M = g.shape[0]
#     if center==True:
#         g -= g.mean(axis=0)
#     v = geweke_c(g=g, s=0)
#     v_L = 0.
#     if L != 0:
#         w = (L-onp.arange(0, L))/L  # weights
#         v *= w[0]
#         assert L > 0 and L < M
#         for s in range(1, L):
#             v_L += w[s] * geweke_c(g=g, s=s)
#     v = (v + 2*v_L)/M
#     return v

# '''
# Calculate the biased `s`-lag autocovariance of *centered* samples `g`
# '''
# def geweke_c(g, s):
#     if s == 0:
#         out = (g ** 2).mean(axis=0)
#     else:
#         M = g.shape[0]
#         out = ((g[s:, :]) * (g[:(M-s), :])).sum(0)/float(M)  # biased
#     return out

'''
Calculate the squared standard error of the estimate of E[`g`] (Geweke 1999, 3.7-8). If `L`=None, automatically selects bandwidth for the lag window based on an asymptotic MSE criterion (Andrews 1991). This assumes that `g` is fourth-moment stationary and the autocovariances are L1-summable.
Note: depends on arch
'''
def geweke_se2(g, L=None, force_int_L=False):
    if len(g.shape) == 1:
        g = g.reshape(-1, 1)
    M = g.shape[0]
    if L is not None:
        bw = max(L-1, 0)
    else:
        bw = None
    v = onp.array([float(Bartlett(g[:, j], bandwidth=bw,
                                  force_int=force_int_L).cov.long_run) for j in range(g.shape[1])])
    v /= M
    return v

'''
Run Geweke test (Geweke 2004) on marginal-conditional and successive-conditional test function arrays `g_mc` and `g_sc`, each row corresponding to a sample.
Uses a maximum window size of `l`*M to estimate of the squared standard error of E[g_sc], where M is the number of SC samples.
Example values of `l` are 0.04, 0.08, 0.15. `l`=None for automatic lag window bandwidth selection (Andrews 1991).
`test_correction` corrects for multiple testing if not set to `test_correction=None`; set to 'b' (for Bonferroni) or 'bh' (for Benjamini-Hochberg) are supported
'''
def geweke_test(g_mc, g_sc, alpha=0.05, l=None, test_correction='bh'):
    if test_correction is not None:
        assert test_correction in ['b', 'bh']
        num_tests = g_mc.shape[1]
    if len(g_mc.shape) == 1 or len(g_sc.shape) == 1:
        g_mc = g_mc.reshape(-1, 1)
        g_sc = g_sc.reshape(-1, 1)
    assert len(g_mc.shape) == 2 and len(g_sc.shape) == 2
    assert g_mc.shape[1] == g_sc.shape[1]
    
    mean_mc = g_mc.mean(axis=0)
    se2_mc = geweke_se2(g_mc, L=0)

    M_sc = float(g_sc.shape[0])
    mean_sc = g_sc.mean(axis=0)
    if l is not None:
        L_sc = l*M_sc
    else:
        L_sc = None
    se2_sc = geweke_se2(g_sc, L=L_sc)

    test_statistic = (mean_mc - mean_sc)/onp.sqrt(se2_mc + se2_sc)
    p_value = 2.*(1-norm.cdf(abs(test_statistic)))
        
    if test_correction == 'b':
        threshold = norm.ppf(1.-alpha/(2.)) # asymptotic
        alpha /= num_tests
        result = p_value <= alpha
    elif test_correction == 'bh':
        threshold = None
        rank = onp.empty_like(p_value)
        rank[onp.argsort(p_value)] = onp.arange(1, len(p_value)+1)
        under = p_value <= rank/num_tests * alpha
        if under.sum() > 0:
          rank_max = rank[under].max()
        else:
          rank_max = 0
        result = rank <= rank_max
    else:
        threshold = norm.ppf(1.-alpha/(2.)) # asymptotic
        result = p_value <= alpha
    
    return {'result': result, 'p_value': p_value, 'test_statistic': test_statistic, 'critical_value': threshold, 'test_correction': test_correction}

'''
Generate Geweke P-P plot (Grosse and Duvenaud 2014) for sample vectors x, y. Can also generate Q-Q plots.
'''
def prob_plot(x, y, plot_type='PP', step = 0.005):
    assert plot_type in ['PP', 'QQ']
    x = onp.sort(x)
    y = onp.sort(y)
    z_min = min(onp.min(x), onp.min(y))
    z_max = max(onp.max(x), onp.max(y))

    ecdf = lambda z, x: (x <= z).sum()/float(len(x))
    if plot_type == 'PP':
        pp_x = [ecdf(z, x) for z in onp.arange(z_min, z_max, step * (z_max-z_min))]
        pp_y = [ecdf(z, y) for z in onp.arange(z_min, z_max, step * (z_max-z_min))]
        plt.plot(pp_x, pp_y, marker='o', color='black', fillstyle='none', linestyle='none')
        plt.plot(pp_x, pp_x, color='black')
    elif plot_type == 'QQ':
        q = onp.arange(0., 1.+step, step)
        qq_x = onp.quantile(x, q)
        qq_y = onp.quantile(y, q)
        plt.plot(qq_x, qq_y, marker='o', color='black', fillstyle='none', linestyle='none')
        plt.plot(qq_x, qq_x, color='black')
    pass

#######################################################################
############################## MMD test ###############################
#######################################################################

'''
Run MMD test on samples with shape (n x p). Depends on shogun
'''
# def mmd_test(samples_p, samples_q, lst_kernels, alpha=0.05, train_test_ratio=1, num_runs=1, num_folds=3):
#     if type(lst_kernels).__name__ != 'list':
#         lst_kernels = [lst_kernels]
#     if len(samples_p.shape) == 1:
#         samples_p = samples_p.reshape(samples_p.shape[0], 1)
#     if len(samples_q.shape) == 1:
#         samples_q = samples_q.reshape(samples_q.shape[0], 1)
    
#     assert samples_p.shape[1] == samples_q.shape[1]

#     features_p = sg.RealFeatures(samples_p.T)
#     features_q = sg.RealFeatures(samples_q.T)

#     if samples_p.shape[0] == samples_q.shape[0]:
#         mmd = sg.LinearTimeMMD(features_p, features_q)
#     else:
#         mmd = sg.QuadraticTimeMMD(features_p, features_q)

#     mmd.set_statistic_type(sg.ST_UNBIASED_FULL) # V-statistic = ST_BIASED_FULL
    
#     # if multiple kernels are provided, will learn which maximizes test power and compute the result on a held-out test set
#     if len(lst_kernels) > 1:
#         mmd.set_train_test_mode(True)
#         mmd.set_train_test_ratio(train_test_ratio)
#         for k in lst_kernels:
#             mmd.add_kernel(k)
#         mmd.set_kernel_selection_strategy(sg.KSM_MAXIMIZE_POWER, num_runs, num_folds, alpha)
#         mmd.select_kernel()
#         learnt_kernel_single = sg.GaussianKernel.obtain_from_generic(mmd.get_kernel())
#         width = learnt_kernel_single.get_width()
#     elif len(lst_kernels) == 1:
#         mmd.set_train_test_mode(False)
#         mmd.set_kernel(lst_kernels[0])
#         width = lst_kernels[0].get_width()

#     test_statistic = mmd.compute_statistic()
#     p_value = mmd.compute_p_value(test_statistic)
#     threshold = mmd.compute_threshold(alpha)
#     result = p_value <= alpha
#     return {'result': result, 'p_value': p_value, 'test_statistic': test_statistic, 'critical_value': threshold, 'kernel_width':width}

'''
Quadratic/Linear time MMD test. No shogun dependency, but slow
'''
def mmd_test(X, Y, kernel=rbf_kernel, alpha=0.05, null_samples=100, kernel_learn_method='median_heuristic', mmd_type='unbiased', rng=None, **kwargs):
    assert X.shape[1] == Y.shape[1] and len(X.shape) == 2 and len(Y.shape) == 2
    assert mmd_type in ['biased', 'unbiased', 'linear']
    assert kernel_learn_method is None or kernel_learn_method in ['median_heuristic']
    if rng is None:
        rng = onp.random.default_rng()

    XY = onp.vstack([X,Y])
    K = kernel(XY, XY, **kwargs)

    if kernel_learn_method == 'median_heuristic':
        assert kernel == rbf_kernel
        K_learn = kernel(X, Y)
        K_learn.learn()
        kernel_param = K_learn.params
        K.set_params(kernel_param)

    if mmd_type == 'linear':
        assert X.shape == Y.shape
        n, p = X.shape
        f_kernel = K.f_kernel

        # Calculate test statistic
        test_statistic, var = mmd_l(X, Y, f_kernel, return_2nd_moment=True)
        var -= test_statistic**2
        scale = onp.sqrt(2.*var/n)
        p_value = norm.sf(test_statistic, scale=scale)
        result = p_value <= alpha
        threshold = norm.ppf(1.-alpha, scale=scale)
    else:
        
        # Calculate null distribution       
        K_XYXY = K.eval()
        n_X, p = X.shape
        n_Y = Y.shape[0]
        null_distr = onp.zeros(null_samples)
        for i in range(null_samples):
            ind = rng.permutation(int(n_X+n_Y))
            ind_X = ind[:n_X]
            ind_Y = ind[n_Y:]

            K_XX = K_XYXY[ind_X, :][:, ind_X]
            K_YY = K_XYXY[ind_Y, :][:, ind_Y]
            K_XY = K_XYXY[ind_X, :][:, ind_Y]
            
            if mmd_type == 'unbiased':
                null_distr[i] = mmd_u(K_XX=K_XX, K_YY=K_YY, K_XY=K_XY)
            elif mmd_type == 'biased':
                null_distr[i] = mmd_v(K_XX=K_XX, K_YY=K_YY, K_XY=K_XY)

        # Calculate test statistic
        K_XX = K_XYXY[:n_X, :][:, :n_X]
        K_YY = K_XYXY[n_X:, :][:, n_X:]
        K_XY = K_XYXY[:n_X, :][:, n_X:]

        if mmd_type == 'unbiased':
            test_statistic = mmd_u(K_XX=K_XX, K_YY=K_YY, K_XY=K_XY)
        elif mmd_type == 'biased':
            test_statistic = mmd_v(K_XX=K_XX, K_YY=K_YY, K_XY=K_XY)    

        threshold = onp.quantile(null_distr, 1.-alpha)
        result = test_statistic >= threshold
        p_value = (null_distr >= test_statistic).mean()
    
    return {'result':result, 'p_value':p_value, 'test_statistic':test_statistic, 'critical_value':threshold, 'kernel_param':K.params}

'''
Generate (squared) Quadratic Time MMD u-statistic
'''
def mmd_u(K_XX, K_YY, K_XY, normalize=True):
    assert K_XY.shape[0] == K_XY.shape[1]
    m, n = K_XY.shape
    if normalize == True:
        if m == n:
            z = n
        else:
            z = m * n / (m + n)
    else:
        z = 1.
    return z*(1./(m*(m-1)) * (K_XX.sum() - onp.diag(K_XX).sum()) + 1./(n*(n-1)) * (K_YY.sum() - onp.diag(K_YY).sum()) - 2.*K_XY.mean())

'''
Generate (squared) Linear Time MMD u-statistic
'''
def mmd_l(X, Y, f_kernel, return_2nd_moment=False):
    assert X.shape == Y.shape
    n, p = X.shape

    h = lambda x_i, y_i, x_j, y_j: f_kernel(x_i, x_j) + f_kernel(y_i, y_j) - f_kernel(x_i, y_j) - f_kernel(x_j, y_i)

    n_2 = int(n/2)
    stat = 0
    second = 0
    for i in range(n_2):
        h_i = h(x_i=X[2*i, :], y_i=Y[2*i, :], x_j=X[2*i+1, :], y_j=Y[2*i+1, :])
        stat += h_i
        second += h_i**2
    stat /= n_2
    second /= n_2

    if return_2nd_moment == True:
        return stat, second
    else:
        return stat

#######################################################################
############################ Wild MMD test ############################
#######################################################################

'''
Generate `k` wild bootstrap processes of length `n` for the Wild MMD test. Returns an (n x k) matrix.
'''
def wb_process(n, k=1, l_n=20, center=True, rng=None):
    if rng is None:
        rng = onp.random.default_rng()
    epsilon = rng.normal(size=(n, k))
    W = onp.sqrt(1-onp.exp(-2/l_n)) * epsilon
    
    for i in range(1, n):
        W[i, :] += W[i-1, :] * onp.exp(-1/l_n)

    if center==True:
        W -= W.mean(0).reshape(1, k)
    return W

'''
Generate wild bootstrapped MMD v-statistic for the Wild MMD test
'''
def mmd_wb(K_XX, K_YY, K_XY, normalize=True, wb_l_n=20, wb_center=True, rng=None):
    if rng is None:
        rng = onp.random.default_rng()
    n_X, n_Y = K_XY.shape
    z = 1.
    if n_X == n_Y:
        if normalize == True:
            z = n_X
        W = wb_process(n_X).reshape(-1, 1)
        return z*(W.T @ (K_XX + K_YY - 2*K_XY) @ W)/(n_X**2)
    else:
        w_X = wb_process(n_X, l_n=wb_l_n, center=wb_center, rng=rng).reshape(-1, 1)
        w_Y = wb_process(n_Y, l_n=wb_l_n, center=wb_center, rng=rng).reshape(-1, 1)
        if normalize == True:
            z = n_X * n_Y / (n_X + n_Y)
        return z*(1./n_X**2 * w_X.T @ K_XX @ w_X + 1./n_Y**2 * w_Y.T @ K_YY @ w_Y - 2./(n_X*n_Y) * w_X.T @ K_XY @ w_Y) 

'''
Generate (squared) MMD v-statistic for the Wild MMD test
'''
def mmd_v(K_XX, K_YY, K_XY, normalize=True):
    n_X, n_Y = K_XY.shape
    z = 1.
    if n_X == n_Y:
        if normalize == True:
            z = n_X
    else:
        if normalize == True:
            z = n_X * n_Y / (n_X + n_Y)
    return z*(K_XX.mean() + K_YY.mean() - 2.*K_XY.mean())

'''
Run Wild MMD test on samples with shape (n x p)
'''
def mmd_wb_test(X, Y, kernel=rbf_kernel, alpha=0.05, null_samples=100, kernel_learn_method='median_heuristic', wb_l_n=20, wb_center=True, rng=None, **kwargs):
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(Y.shape[0], 1)
    if rng is None:
        rng = onp.random.default_rng()

    K_XX = kernel(X, X, **kwargs)
    K_YY = kernel(Y, Y, **kwargs)
    K_XY = kernel(X, Y, **kwargs)
    if kernel_learn_method == 'median_heuristic':
        assert kernel == rbf_kernel
        K_learn = kernel(X, Y)
        K_learn.learn()
        kernel_param = K_learn.params
        K_XX.set_params(kernel_param)
        K_YY.set_params(kernel_param)
        K_XY.set_params(kernel_param)

    K_XX_eval = K_XX.eval()
    K_YY_eval = K_YY.eval()
    K_XY_eval = K_XY.eval()

    B = onp.empty(null_samples)
    for i in range(null_samples):
        B[i] = mmd_wb(K_XX_eval, K_YY_eval, K_XY_eval, normalize=True, wb_l_n=wb_l_n, wb_center=wb_center, rng=rng)

    threshold = onp.quantile(B, 1.-alpha)
    test_statistic = mmd_v(K_XX_eval, K_YY_eval, K_XY_eval, normalize=True)
    result = test_statistic >= threshold
    p_value = (B >= test_statistic).mean() # one-sided
    
    return {'result':result, 'p_value':p_value, 'test_statistic':test_statistic, 'critical_value':threshold, 'kernel_param':K_XX.params}

'''
Estimate MMD variance
'''
def mmd_var(K_XX, K_YY, K_XY):
    assert K_XY.shape[0] == K_XY.shape[1]

    n = K_XY.shape[0]
    K_XY_sum = K_XY.sum()

    K_XX_tilde = K_XX.copy()
    onp.fill_diagonal(K_XX_tilde, 0.)
    K_XX_tilde_sum = K_XX_tilde.sum()

    K_YY_tilde = K_YY.copy()
    onp.fill_diagonal(K_YY_tilde, 0.)
    K_YY_tilde_sum = K_YY_tilde.sum()

    var = 4/perm(n, 4) * (onp.linalg.norm(K_XX_tilde.sum(1))**2 + onp.linalg.norm(K_YY_tilde.sum(1))**2) + \
        4*(n**2-n-1)/(n**3*(n-1)**2) * (onp.linalg.norm(K_XY.sum(0))**2 + onp.linalg.norm(K_XY.sum(1))**2) + \
        -8/(n**2*(n**2-3*n+2)) * ((K_XX_tilde @ K_XY).sum() + (K_XY @ K_YY_tilde).sum()) + \
        8/(n**2*perm(n, 3)) * (K_XX_tilde_sum + K_YY_tilde_sum) * K_XY_sum + \
        -2*(2*n-3)/(perm(n, 2)*perm(n, 4))*(K_XX_tilde_sum**2 + K_YY_tilde_sum**2) + \
        -4*(2*n-3)/(n**3*(n-1)**3) * K_XY_sum**2 + \
        -2/(n*(n**3-6*n**2+11*n-6)) * (onp.linalg.norm(K_XX_tilde)**2 + onp.linalg.norm(K_YY_tilde)**2) + \
        4*(n-2)/(n**2*(n-1)**3)*onp.linalg.norm(K_XY)**2

    return var
  
# From Sutherland et al. 2016. For checking
def mmd_var_alt(K_XX, K_XY, K_YY):
    m = K_XX.shape[0]

    diag_X = onp.diag(K_XX)
    diag_Y = onp.diag(K_YY)

    sum_diag_X = diag_X.sum()
    sum_diag_Y = diag_Y.sum()

    sum_diag2_X = diag_X.dot(diag_X)
    sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
    K_XY_2_sum  = (K_XY ** 2).sum()


    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              K_XY_sums_1.dot(K_XY_sums_1)
            + K_XY_sums_0.dot(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
    )

    return var_est


'''
Learn RBF kernel width for Wild Bootstrap MMD by maximizing the (unbiased) MMD t-statistic
`lst_tau` is a list of candidate widths, and `X` and `Y` are (n x p) *training* samples
'''
def learn_kernel_rbf(lst_tau, X, Y, alpha=0.05):
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(Y.shape[0], 1)

    assert X.shape == Y.shape
    n = X.shape[0]
    min_variance = 1e-8
    
    # Maximize t-stat
    lst_objective = [None]*len(lst_tau)
    objective_max = -onp.inf
    for i, tau in enumerate(lst_tau):
        f_kernel = lambda X, Y: rbf_kernel(X, Y, tau)
        K_XX = f_kernel(X, X)
        K_YY = f_kernel(Y, Y)
        K_XY = f_kernel(X, Y)
        
        test_statistic = mmd_u(K_XX, K_YY, K_XY, normalize=False)
        variance = mmd_var(K_XX, K_YY, K_XY)
        objective = test_statistic/onp.sqrt(max(variance, min_variance))
        lst_objective[i] = objective
        if objective >= objective_max:
            objective_max = objective
            tau_opt = tau
    
    return tau_opt, lst_tau, lst_objective



#######################################################################
#################### Tests from Gandy & Scott 2020 ##################
#######################################################################

def ks_test(X, Y, alpha=0.05):
    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[1] == Y.shape[1]
    p_values = onp.array([ks_2samp(X[:, j], Y[:, j]).pvalue for j in range(X.shape[1])])
    result = (p_values <= alpha/X.shape[1]) # Bonferroni correction  
    return {'result': result, 'p_value': p_values}

def rank_stat(model, L, rng=None):
    if rng is None:
        rng = onp.random.default_rng()
    
    M = rng.choice(L)

    chain = onp.zeros(shape=(L, len(model.theta_indices)))
    chain[M, :] = model.drawPrior()

    y = model.drawLikelihood()
    
    stateDict = model.__dict__.copy()

    # Backward
    for i in range(M-1, -1, -1):
        chain[i, :] = model.drawPosterior()

    # Forward
    model.__dict__ = stateDict.copy()
    for j in range(M+1, L):
        chain[j, :] = model.drawPosterior()
    
    # Apply test functions
    chain = onp.hstack([onp.repeat(y.reshape(1, model._N), repeats=L, axis=0), chain])
    chain = model.test_functions(chain)
    return rankdata(chain, 'ordinal', axis = 0)[M, :]

def rank_test(model, N, L, alpha=0.05, rng=None):
    if rng is None:
        rng = onp.random.default_rng()

    ranks = onp.vstack([rank_stat(model=model, L=L) for _ in range(N)])
    f_obs = onp.apply_along_axis(lambda x: onp.bincount(x, minlength=L), axis=0, arr=ranks-1)
    p_values = onp.array([chisquare(f_obs[:, j]).pvalue for j in range(ranks.shape[1])])
    
    result = (p_values <= alpha/ranks.shape[1]) # Bonferroni correction  
    return {'result': result, 'p_value': p_values}    

# Sequential wrapper
def f_test_sequential(sample_size, model, test_type, **kwargs):
    assert test_type in ['rank', 'ks', 'mmd', 'mmd_sum_uni', 'mmd-wb', 'geweke']
        
    if test_type == 'rank':
        p_values = rank_test(model, N=500, L=5)['p_value']
    elif test_type in ['ks', 'mmd', 'mmd_sum_uni']:
        X = model.test_functions(model.sample_mc(sample_size))
        Y = model.test_functions(model.sample_bc(sample_size, burn_in_samples=5))
        if test_type == 'ks':
            p_values = ks_test(X, Y)['p_value']
        elif test_type == 'mmd':
            p_values = mmd_test(X, Y, kernel=rbf_kernel, mmd_type='unbiased')['p_value']
        elif test_type == 'mmd_sum_uni':
            p_values = mmd_test(X, Y, kernel=sum_uni_rbf_kernel, mmd_type='unbiased')['p_value']
    elif test_type in ['mmd-wb', 'geweke']:
        if test_type == 'mmd-wb':
            mmd_test_size = int(sample_size)
            mmd_thinning = onp.arange(0, int(sample_size), 1) #int(sample_size/mmd_test_size)
            X = model.test_functions(model.sample_mc(mmd_test_size))
            Y = model.test_functions(model.sample_sc(sample_size))
            p_values = mmd_wb_test(X, Y[mmd_thinning, :])['p_value']
        elif test_type == 'geweke':
            geweke_thinning = onp.arange(0, int(sample_size), 1)
            X = model.test_functions(model.sample_mc(sample_size))
            Y = model.test_functions(model.sample_sc(sample_size))
            p_values = geweke_test(X, Y[geweke_thinning, :], l=0.15, test_correction='b')['p_value']        
    return p_values
    
def sequential_test(f_test, n, alpha, k, Delta):
    beta = alpha/k
    gamma = beta**(1/k)

    for i in range(k):
        p = f_test(n)
        if type(p).__name__ == 'ndarray':
            d = p.flatten().shape[0]
            q = onp.min(p)*d
        else:
            q = p
        
        if onp.isnan(q):
            return True
        if q <= beta:
            return True
        if q > gamma + beta:
            break
        beta = beta/gamma

        if i == 0:
            n = Delta * n
    return False

#######################################################################
############################# Experiments #############################
#######################################################################
'''
Generic class to run multiple trials of an experiment in parallel
'''
class experiment(object):
    def __init__(self, num_trials, nproc, seed=None):
        self._num_trials = num_trials
        self._nproc = nproc
        if seed is None:
            self._seed_sequence = onp.random.SeedSequence()
        else:
            self._seed = seed
            self._seed_sequence = onp.random.SeedSequence(self._seed)
        self._child_seed_sequences = self._seed_sequence.spawn(self._nproc)
        pass

    def run_trial(self):
        pass

    def save_trial(self, res, trial):
        pass

    def run_experiment(self, num_trials, seed_sequence):
        # set the seed of the sampler based on the passed seed sequence
        seed = seed_sequence.generate_state(1)
        self._sampler.set_seed(seed)

        lst_paths = [None] * num_trials
        for trial in range(num_trials):
          res = self.run_trial()
          lst_paths[trial] = self.save_trial(res, trial)
          del res
        return lst_paths

    def run(self):
        print(f'Running {self._num_trials} trials with {self._nproc} processes')
        if self._nproc == 1:
            out = self.run_experiment(self._num_trials, self._child_seed_sequences[0])
        else:
            lst_num_trials = splitIter(int(self._num_trials), int(self._nproc))
            pool = multiprocessing.Pool(processes=int(self._nproc))
            out = pool.starmap(self.run_experiment, zip(lst_num_trials, self._child_seed_sequences))
            pool.close()
        return out

'''
Experiment class for comparing various marginal samples of theta using Geweke, MMD-backward, and wild-MMD-Geweke joint tests. `tau` is the bandwidth for the MMD kernel
'''
class joint_test_experiment(experiment):
    def __init__(self, num_trials, nproc, seed=None, **kwargs):
        super().__init__(num_trials, nproc, seed)
        for key, value in kwargs.items():
            setattr(self, '_' + key, value)
        for attr in ['_experiment_name', '_sampler', '_num_trials', '_nproc', '_num_samples', '_burn_in_samples_bc', '_geweke_thinning_samples', '_mmd_thinning_samples', '_tau', '_alpha', '_l_geweke']:
            assert hasattr(self, attr)

        self._dir_tests = './results/' + \
            type(self._sampler).__name__.replace('_sampler', '') + '/'
        self._dir_samples = './samples/' + \
            type(self._sampler).__name__.replace('_sampler', '') + '/'
        if not os.path.isdir(self._dir_tests):
            os.makedirs(self._dir_tests)
        if not os.path.isdir(self._dir_samples):
            os.makedirs(self._dir_samples)
        pass

    def save_trial(self, res, trial):
        file_name = '/experiment_' + str(self._experiment_name) + '_' + \
            str(os.getpid()) + '_' + str(trial) + '.pkl'
        
        lst_paths = [None] * 2
        for i, d in enumerate([self._dir_tests, self._dir_samples]):
            path = d+file_name
            with open(path, 'wb') as f:
                pickle.dump(res[i], f)
            lst_paths[i] = path

        return lst_paths

    def run_trial(self):
        theta_indices = self._sampler.theta_indices
        thinned_samples_geweke = onp.arange(
            0, self._num_samples, self._geweke_thinning_samples).astype('int')  # thinning for Geweke test
        thinned_samples_mmd = onp.arange(0, self._num_samples, self._mmd_thinning_samples).astype(
            'int')  # thinning for MMD tests

        samples_p = self._sampler.sample_mc(self._num_samples)
        samples_q = self._sampler.sample_bc(
            int(self._num_samples / self._mmd_thinning_samples), burn_in_samples=self._burn_in_samples_bc)
        samples_r = self._sampler.sample_sc(self._num_samples)

        # Geweke test
        time_start_geweke = perf_counter()
        tests_geweke = geweke_test(geweke_functions(samples_p[thinned_samples_geweke, :][:, theta_indices]), geweke_functions(
            samples_r[thinned_samples_geweke, :][:, theta_indices]), l=self._l_geweke, alpha=self._alpha)
        time_end_geweke = perf_counter()

        # backward MMD test
        time_start_backward = perf_counter()
        tests_backward = mmd_test(samples_p[thinned_samples_mmd, :][:, theta_indices],
                                  samples_q[:, theta_indices], sg.GaussianKernel(10, self._tau), alpha=self._alpha)
        time_end_backward = perf_counter()

        # wild MMD-SC test
        time_start_wild = perf_counter()
        def f_kernel(X, Y): return rbf_kernel(X, Y, tau=self._tau)
        tests_wild = mmd_wb_test(samples_p[thinned_samples_mmd, :][:, theta_indices],
                                 samples_r[thinned_samples_mmd, :][:, theta_indices], f_kernel, alpha=self._alpha)
        time_end_wild = perf_counter()

        test_runtimes = {'geweke': time_end_geweke-time_start_geweke,
                         'backward': time_end_backward-time_start_backward, 'wild': time_end_wild-time_start_wild}

        tests = {'geweke': tests_geweke, 'backward': tests_backward, 'wild': tests_wild,
                 'specification': self.__dict__, 'test_runtimes': test_runtimes}
        samples = {'mc': samples_p, 'bc': samples_q, 'sc': samples_r}
        return tests, samples


'''
Generate samples in parallel
'''
class parallel_sampler(experiment):
    def __init__(self, num_trials, nproc, seed=None, **kwargs):
        super().__init__(num_trials, nproc, seed)
        for key, value in kwargs.items():
            setattr(self, '_' + key, value)
        for attr in ['_experiment_name', '_sampler', '_num_trials', '_nproc', '_num_samples_mc', '_num_samples_sc', '_num_samples_bc', '_burn_in_samples_bc']:
            assert hasattr(self, attr)
        
        self._sampler.set_nproc(1)

        self._dir_samples = './samples/' + \
            type(self._sampler).__name__.replace('_sampler', '') + '/'
        if not os.path.isdir(self._dir_samples):
            os.makedirs(self._dir_samples)
        pass

    def save_trial(self, res, trial):
        file_name = '/experiment_' + str(self._experiment_name) + '_' + \
            str(os.getpid()) + '_' + str(trial) + '.pkl'
        path = self._dir_samples  + file_name
        with open(path, 'wb') as f:
            pickle.dump(res, f)
        return path

    def run_trial(self):
        theta_indices = self._sampler.theta_indices
        samples_p = self._sampler.sample_mc(self._num_samples_mc)
        samples_q = self._sampler.sample_bc(self._num_samples_bc, self._burn_in_samples_bc)
        samples_r = self._sampler.sample_sc(self._num_samples_sc)

        samples = {'mc': samples_p, 'bc': samples_q, 'sc': samples_r}
        return samples
