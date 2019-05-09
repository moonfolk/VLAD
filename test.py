import numpy as np
from vlad import vlad, gdm
from evaluate import gen_gauss_nmf, evaluate

def gen_data(M, alpha, scale_min, D, K):
    M_test = 1000
    X, beta_t, _, _, _, X_test = gen_gauss_nmf(M, D, K, M_test, alpha, scale_min=scale_min, scale_max=1.)
    return X, K, beta_t, X_test

np.random.seed(1)
M = 10000
scale_min = 0.5
D = 500
K = 10
alpha=1.

# Generate data
X, K, beta_t, X_test = gen_data(M, alpha, scale_min, D, K)

# Run VLAD
gam_beta = vlad(X, K, it=2000, n_jobs=4, n_init=8, deconv=True, alpha_m_path='./alpha_m_%d.npy' % K)
print('VLAD MM distance %f; NMF objective %f\n' % evaluate(X_test, gam_beta, beta_t, alpha=alpha))

# Run VLAD with true alpha
gam_beta = vlad(X, K, it=2000, n_jobs=4, n_init=8, deconv=True, alpha=alpha)
print('VLAD with true alpha MM distance %f; NMF objective %f\n' % evaluate(X_test, gam_beta, beta_t, alpha=alpha))

# Run GDM
gam_beta = gdm(X, K, n_jobs=4, n_init=8)
print('GDM MM distance %f; NMF objective %f\n' % evaluate(X_test, gam_beta, beta_t, alpha=alpha))