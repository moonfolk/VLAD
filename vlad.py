import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
    
def simul_m(K, alpha, M=100000):
    X = np.random.dirichlet(np.ones(K)*alpha, M)
    kmeans = KMeans(n_clusters=K, n_jobs=1, n_init=1, max_iter=5000).fit(X)
    centers = kmeans.cluster_centers_
    
    cent = 1./K * np.ones(K)
    m = np.sqrt(K*(K-1.))/norm(centers - cent, axis=1).sum()
    return np.ones(K)*m


def get_beta(cent, centers, m, lda=False, pois=False):
    betas = np.array([cent + m[x]*(centers[x,:] - cent) for x in range(centers.shape[0])])
    if lda:
        betas[betas<0] = 0
        betas = normalize(betas, 'l1')
    if pois:
        betas[betas<0] = 0
        
    return betas

def gdm(X, K, n_jobs=1, n_init=10, lda=False, pois=False):
    cent = np.mean(X, axis=0)
    kmeans = KMeans(n_clusters=K, n_jobs=n_jobs, n_init=n_init).fit(X)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    m = []
    for k in range(K):
        k_dist = euclidean(cent, centers[k])
        Rk = max(np.apply_along_axis(lambda x: euclidean(cent, x), 1, X[labels==k,:]))
        m.append(Rk/k_dist)
    
    beta = get_beta(cent, centers, m, lda=lda, pois=pois)

    return beta

def match_covariance_gaus(X, gam_centers, alpha_range, alpha_m, sigma2_deconv):
    
    K, D = gam_centers.shape
    x_mean = X.mean(axis=0)
    x_cov = np.dot((X-x_mean).T, X-x_mean)/X.shape[0]
    P = -np.ones((K,K)) + np.eye(K)*K
    obj_value = []
    for a,m in zip(alpha_range,alpha_m):
        alpha_const = 1./(K**2*(K*a+1))
        beta = get_beta(x_mean, gam_centers, np.ones(K)*m, lda=False, pois=False)
        est_cov = alpha_const*np.dot(np.dot(beta.T,P),beta) + np.eye(D)*sigma2_deconv
        obj_value.append(norm(x_cov-est_cov))
    
    print('Estimated alpha %f\n' % (alpha_range[np.argmin(obj_value)]))
    return alpha_m[np.argmin(obj_value)]*np.ones(K)

def match_covariance_pois(X, gam_centers, alpha_range, alpha_m):
    
    K, D = gam_centers.shape
    x_mean = X.mean(axis=0)
    x_cov = np.dot((X-x_mean).T, X-x_mean)/X.shape[0]
    P = -np.ones((K,K)) + np.eye(K)*K
    obj_value = []
    for a,m in zip(alpha_range,alpha_m):
        alpha_const = 1./(K**2*(K*a+1))
        beta = get_beta(x_mean, gam_centers, np.ones(K)*m, lda=False, pois=False)
        est_cov = alpha_const*np.dot(np.dot(beta.T,P),beta) - np.diag(beta.mean(axis=0))
        obj_value.append(norm(x_cov-est_cov))
    
    print('Estimated alpha %f\n' % (alpha_range[np.argmin(obj_value)]))
    return alpha_m[np.argmin(obj_value)]*np.ones(K)

def match_covariance_lda(X, gam_centers, alpha_range, alpha_m, Nm):

    K, D = gam_centers.shape
    x_mean = X.mean(axis=0)
    x_cov = np.dot((X-x_mean).T, X-x_mean)/X.shape[0]
    P = -np.ones((K,K)) + np.eye(K)*K
    obj_value = []
    for a,m in zip(alpha_range,alpha_m):
        alpha_const = 1./(K**2*(K*a+1))
        beta = get_beta(x_mean, gam_centers, np.ones(K)*m, lda=False, pois=False)
        est_cov = alpha_const*np.dot(np.dot(beta.T,P),beta)*(1-1./Nm) + np.diag(beta.mean(axis=0))/Nm - np.outer(beta.mean(axis=0),beta.mean(axis=0))/Nm
        obj_value.append(1e6*norm(x_cov-est_cov))

    print('Estimated alpha %f\n' % (alpha_range[np.argmin(obj_value)]))
    return alpha_m[np.argmin(obj_value)]*np.ones(K)
        
def vlad(X, K, alpha=None, it=1000, n_jobs=1, n_init=10, deconv=True, lda=False, pois=False, alpha_m_path=None, Nm=None):
    
    cent = X.mean(axis=0)        
        
    u, s, v = np.linalg.svd(X-cent, full_matrices=False)
    
    if deconv:
        sigma2_deconv = s[K]**2/X.shape[0]
        s = np.sqrt(s[:K]**2 - s[K]**2)
    else:
        s = s[:K]

    X_svd = u[:,:K]
    
    v_back = s.reshape(-1,1)*v[:K]
    
    if lda:
        init_centers = KMeans(n_clusters=K, n_jobs=n_jobs, n_init=n_init, max_iter=10).fit(X-cent).cluster_centers_
        v_to_svd = v[:K].T * 1./s
        init_centers = np.dot(init_centers, v_to_svd)
        u_centers = KMeans(n_clusters=K, init=init_centers, max_iter=it, n_init=1).fit(X_svd).cluster_centers_
    else:
        u_centers = KMeans(n_clusters=K, n_init=n_init, max_iter=it, n_jobs=n_jobs).fit(X_svd).cluster_centers_
    
    D_centers = np.dot(u_centers, v_back) + cent
    
    if alpha is None:    
        alpha_range = np.arange(0.05,6.01,0.01)
        if alpha_m_path is None:
            print('Make grid for learning alpha - this may take some time. Needs to be done once for a fixed K (independent of data)')
            from joblib import Parallel, delayed
            alpha_m = Parallel(n_jobs=n_jobs)(delayed(simul_m)(K, a) for a in alpha_range)
            alpha_m = [a_m[0] for a_m in alpha_m]
            np.save('alpha_m_%d' % K, np.array(alpha_m))
            print('Done making the grid and saved it - reuse on next run')
        else:
            alpha_m = np.load(alpha_m_path)
        if deconv:
            m = match_covariance_gaus(X, D_centers, alpha_range, alpha_m, sigma2_deconv)
        elif pois:
            m = match_covariance_pois(X, D_centers, alpha_range, alpha_m)
        elif lda:
            m = match_covariance_lda(X, D_centers, alpha_range, alpha_m, Nm)
        else:
            print('Unknown distribution')


    else:
        m = simul_m(K, alpha)
    
    beta = get_beta(cent, D_centers, m, lda=lda, pois=pois)
    
    return beta