import multiprocessing
import numpy as onp
import shogun as sg
from matplotlib import pyplot as plt
from scipy.stats import norm
import pickle
from time import perf_counter

''' 
Split `num_iter` iterations into `nthreads` chunks for multithreading
'''
def splitIter(num_iter, nthreads):
    arr_iter = (onp.zeros(nthreads) + num_iter // nthreads).astype('int')
    for i in range(num_iter % nthreads):
        arr_iter[i] += 1
    return arr_iter

''' 
Generate RBF kernel matrix with width `tau`. 
Each row corresponds to an observation (row) from X; each column corresponds to an observation (row) from Y.
'''
def rbf_kernel(X, Y, tau):
    n_X, p = X.shape
    n_Y, p_Y = Y.shape
    assert p == p_Y
    K = onp.exp(-((X.reshape(n_X, 1, p) - Y.reshape(1, n_Y, p))**2).sum(2) / tau)
    return K

#######################################################################
############################# Geweke test #############################
#######################################################################

'''
Calculate the squared standard error of the estimate of E[`g`] (Geweke 1999, 3.7-8).
'''
def geweke_se2(g, L=None, center=True):
    M = g.shape[0]
    if center==True:
        g -= g.mean(axis=0)
    v = geweke_c(g=g, s=0)
    v_L = 0.
    if L is not None:
        L = int(L)
        assert L > 0 and L < M
        for s in range(1, L):
            v_L += (L-s)/L * geweke_c(g=g, s=s)
    v = (v + 2*v_L)/M
    return v

'''
Calculate the biased `s`-lag autocovariance of *centered* test functions `g`
'''
def geweke_c(g, s):
    if s==0:
        out = (g ** 2).mean(axis = 0)
    else:
        M = g.shape[0]
        out = ((g[s:, :]) * (g[:(M-s), :])).sum(0)/float(M) # biased
    return out

'''
Run Geweke test (Geweke 2004) on marginal-conditional and successive-conditional test function arrays `g_mc` and `g_sc`, each row corresponding to a sample.
Uses a maximum window size of `l`*M to estimate of the squared standard error of E[g_sc], where M is the number of SC samples.
Example values of `l` are 0.04, 0.08, 0.15.
'''
def geweke_test(g_mc, g_sc, alpha, l=0.15, use_bonferroni=False):
    if len(g_mc.shape) == 1 or len(g_sc.shape) == 1:
        g_mc = g_mc.reshape(-1, 1)
        g_sc = g_sc.reshape(-1, 1)
    assert len(g_mc.shape) == 2 and len(g_sc.shape) == 2
    assert g_mc.shape[1] == g_sc.shape[1]
    
    mean_mc = g_mc.mean(axis=0)
    se2_mc = geweke_se2(g_mc)

    M_sc = float(g_sc.shape[0])
    mean_sc = g_sc.mean(axis=0)
    if l == 0:
        L_sc = None
    else:
        L_sc = l*M_sc
    se2_sc = geweke_se2(g_sc, L=L_sc)
    
    if use_bonferroni==True:
        m = g_mc.shape[1]
    else:
        m = 1.
    
    threshold = norm.ppf(1.-alpha/(2.*m)) # asymptotic
    test_statistic = (mean_mc - mean_sc)/onp.sqrt(se2_mc + se2_sc)
    
    result = abs(test_statistic) >= threshold
    p_value = 2.*(1-norm.cdf(abs(test_statistic)))
    return {'result': result, 'p_value': p_value, 'test_statistic': test_statistic, 'critical_value': threshold}

'''
Returns a matrix with column means corresponding to the first and second empirical moments of `samples`.
'''
def geweke_functions(samples):
    f1 = samples.copy()
    n, p = f1.shape
    f2 = onp.empty([n, int(p*(p+1)/2)])
    counter = 0
    for i in range(p):
        for j in range(i+1):
            f2[:, counter] = f1[:, i] * f1[:, j]
            counter += 1
    return onp.hstack([f1, f2])

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
        # plt.title('PP Plot')
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
Run MMD test on samples with shape (n x p)
'''
def mmd_test(samples_p, samples_q, lst_kernels, alpha=0.05, train_test_ratio=1, num_runs=1, num_folds=3):
    if type(lst_kernels).__name__ != 'list':
        lst_kernels = [lst_kernels]
    if len(samples_p.shape) == 1:
        samples_p = samples_p.reshape(samples_p.shape[0], 1)
    if len(samples_q.shape) == 1:
        samples_q = samples_q.reshape(samples_q.shape[0], 1)

    features_p = sg.RealFeatures(samples_p.T)
    features_q = sg.RealFeatures(samples_q.T)

    mmd = sg.QuadraticTimeMMD(features_p, features_q)
    mmd.set_statistic_type(sg.ST_UNBIASED_FULL) # V-statistic = ST_BIASED_FULL
    
    # if multiple kernels are provided, will learn which maximizes test power and compute the result on a held-out test set
    if len(lst_kernels) > 1:
        mmd.set_train_test_mode(True)
        mmd.set_train_test_ratio(train_test_ratio)
        for k in lst_kernels:
            mmd.add_kernel(k)
        mmd.set_kernel_selection_strategy(sg.KSM_MAXIMIZE_POWER, num_runs, num_folds, alpha)
        mmd.select_kernel()
        learnt_kernel_single = sg.GaussianKernel.obtain_from_generic(mmd.get_kernel())
        width = learnt_kernel_single.get_width()
    elif len(lst_kernels) == 1:
        mmd.set_train_test_mode(False)
        mmd.set_kernel(lst_kernels[0])
        width = lst_kernels[0].get_width()

    test_statistic = mmd.compute_statistic()
    p_value = mmd.compute_p_value(test_statistic)
    threshold = mmd.compute_threshold(alpha)
    result = p_value <= alpha
    return {'result': result, 'p_value': p_value, 'test_statistic': test_statistic, 'critical_value': threshold, 'kernel_width':width}


#######################################################################
############################ Wild MMD test ############################
#######################################################################

'''
Generate `k` wild bootstrap processes of length `n` for the Wild MMD test. Returns an (n x k) matrix.
'''
def wb_process(n, k=1, l_n=20, rng=None, center=True):
    if rng is None:
        rng = onp.random.default_rng()
    epsilon = rng.normal(size=(n, k))
    W = onp.sqrt(1-onp.exp(-2/l_n)) * epsilon
    
    for i in range(1, n):
        W[i, :] = W[i-1, :] * onp.exp(-1/l_n) + W[i, :]

    if center==True:
        W-=W.mean()
    return W

'''
Generate wild bootstrapped MMD v-statistic for the Wild MMD test
'''
def mmd_wb(K_XX, K_YY, K_XY, rng=None):
    if rng is None:
        rng = onp.random.default_rng()
    n_X, n_Y = K_XY.shape
    if n_X == n_Y:
        W = wb_process(n_X).reshape(-1, 1)
        return (W.T @ (K_XX + K_YY - 2*K_XY) @ W)/n_X
    else:
        n = n_X + n_Y
        rho_x = n_X/n
        rho_y = n_Y/n
        w_X = wb_process(n_X, rng=rng).reshape(-1, 1)
        w_Y = wb_process(n_Y, rng=rng).reshape(-1, 1)
        return rho_x*rho_y*n*(1./n_X**2 * w_X.T @ K_XX @ w_X + 1./n_Y**2 * w_Y.T @ K_YY @ w_Y - 2./(n_X*n_Y) * w_X.T @ K_XY @ w_Y) 

'''
Generate MMD v-statistic for the Wild MMD test
'''
def mmd_v(K_XX, K_YY, K_XY):
    n_X, n_Y = K_XY.shape
    if n_X == n_Y:
        return n_X * (K_XX + K_YY - 2*K_XY).mean()
    else:    
        n = n_X + n_Y
        rho_x = n_X/n
        rho_y = n_Y/n
        return rho_x*rho_y*n*(1./n_X**2 * K_XX.sum() + 1./n_Y**2 * K_YY.sum() - 2./(n_X*n_Y) * K_XY.sum())

'''
Run Wild MMD test on samples with shape (n x p)
'''
def mmd_wb_test(X, Y, f_kernel, alpha, null_samples=100, rng=None):
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(Y.shape[0], 1)
    if rng is None:
        rng = onp.random.default_rng()
    
    K_XX = f_kernel(X, X)
    K_YY = f_kernel(Y, Y)
    K_XY = f_kernel(X, Y)
    
    B = onp.empty(null_samples)
    for i in range(null_samples):
        B[i] = mmd_wb(K_XX, K_YY, K_XY, rng=rng)

    threshold = onp.quantile(B, 1.-alpha)
    test_statistic = mmd_v(K_XX, K_YY, K_XY)
    result = test_statistic >= threshold
    p_value = (B >= test_statistic).mean() # one-sided
    
    return {'result':result, 'p_value':p_value, 'test_statistic':test_statistic, 'critical_value':threshold}

#######################################################################
############################# Experiments #############################
#######################################################################
'''
Generic class to run multiple trials of an experiment in parallel
'''
class experiment(object):
    def __init__(self, num_trials, nthreads, seed=None):
        self._num_trials = num_trials
        self._nthreads = nthreads
        if seed is None:
            self._seed_sequence = onp.random.SeedSequence()
        else:
            self._seed = seed
            self._seed_sequence = onp.random.SeedSequence(self._seed)
        self._child_seed_sequences = self._seed_sequence.spawn(self._nthreads)
        pass

    def run_trial(self):
        pass

    def run_experiment(self, num_trials, **kwargs):
        lst_results = [None] * num_trials
        for trial in range(num_trials):
            lst_results[trial] = self.run_trial()
        return lst_results

    def run(self):
        print(f'Running {self._num_trials} trials with {self._nthreads} threads')
        if self._nthreads == 1:
            out = self.run_experiment(self._num_trials, self._child_seed_sequences[0])
        else:
            lst_num_trials = splitIter(int(self._num_trials), int(self._nthreads))
            pool = multiprocessing.Pool(processes=int(self._nthreads))
            out = pool.starmap(self.run_experiment, zip(lst_num_trials, self._child_seed_sequences, onp.arange(len(lst_num_trials))))
            pool.close()
        return out

'''
Sampling experiment class for comparing various marginal samples of theta using Geweke, MMD-backward, and wild-MMD-Geweke tests
'''
class sample_experiment(experiment):
    def __init__(self, num_trials, nthreads, seed=None, **kwargs):
        super().__init__(num_trials, nthreads, seed)
        for key, value in kwargs.items():
            setattr(self, '_' + key, value)
        for attr in ['_sampler', '_num_trials', '_nthreads', '_num_samples', '_burn_in_samples', '_geweke_thinning_samples', '_mmd_thinning_samples', '_tau', '_alpha', '_savedir']:
            assert hasattr(self, attr)
        pass

    def run_experiment(self, num_trials, seed_sequence, experiment_id):
        seed = seed_sequence.generate_state(1)
        self._sampler.set_seed(seed) # set the seed of the sampler based on the passed seed sequence

        # lst_results = super().run_experiment(num_trials)
        
        lst_paths = []
        for trial in range(num_trials):
          lst_results = super().run_experiment(1)
          path = self._savedir + '/' + str(self._experiment_name) +'_results_' + str(experiment_id) + '_' + str(trial) + '.pkl'
          with open(path, 'wb') as f:
            pickle.dump(lst_results, f)
          del lst_results
          lst_paths += path
        return lst_paths

    def run_trial(self):
        theta_indices = self._sampler.theta_indices
        thinned_samples_geweke = onp.arange(0, self._num_samples, self._geweke_thinning_samples).astype('int') # thinning for Geweke test
        thinned_samples_mmd = onp.arange(0, self._num_samples, self._mmd_thinning_samples).astype('int') # thinning for MMD tests

        samples_p = self._sampler.sample_mc(self._num_samples)
        samples_q = self._sampler.sample_bc(
            int(self._num_samples / self._mmd_thinning_samples), burn_in_samples=self._burn_in_samples)
        samples_r = self._sampler.sample_sc(self._num_samples)
        
        # Geweke test
        time_start_geweke = perf_counter()
        tests_geweke = geweke_test(geweke_functions(samples_p[thinned_samples_geweke, :][:, theta_indices]), geweke_functions(
            samples_r[thinned_samples_geweke, :][:, theta_indices]), l=0.15, alpha=self._alpha)
        time_end_geweke = perf_counter()
        
        # backward MMD test
        time_start_backward = perf_counter()
        tests_backward = mmd_test(samples_p[thinned_samples_mmd, :][:,theta_indices],samples_q[:, theta_indices], sg.GaussianKernel(10, self._tau), alpha=self._alpha)
        time_end_backward = perf_counter()
        
        # wild MMD-SC test
        time_start_wild = perf_counter()
        f_kernel = lambda X, Y: rbf_kernel(X, Y, tau=self._tau)
        tests_wild = mmd_wb_test(samples_p[thinned_samples_mmd, :][:, theta_indices], samples_r[thinned_samples_mmd, :][:, theta_indices], f_kernel, alpha=self._alpha)
        time_end_wild = perf_counter()
        
        test_runtimes = {'geweke':time_end_geweke-time_start_geweke, 'backward':time_end_backward-time_start_backward, 'wild':time_end_wild-time_start_wild}
        
        tests = {'geweke': tests_geweke, 'backward': tests_backward, 'wild': tests_wild, 'specification': self.__dict__, 'test_runtimes':test_runtimes}
        samples = {'mc': samples_p, 'bc': samples_q, 'sc': samples_r}
        return tests, samples
