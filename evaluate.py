import numpy as np
from numpy.linalg import lstsq, norm

## Data generation
def gen_gauss_nmf(M=5000, D=500, K=10, M_test = 1000, alpha=0.1, mean=0., sd_beta=1., sd_x=1., scale_min=0.1, scale_max=0.4):
    beta_t = np.random.normal(mean, sd_beta*np.sqrt(K), (K,D))
    theta_t = np.random.dirichlet(np.ones(K)*alpha, M)
    theta_test = np.random.dirichlet(np.ones(K)*alpha, M_test)
    
    c_scale = np.random.uniform(scale_min,scale_max,size=K)
    for i,c in enumerate(c_scale):
        beta_t[i] *= c
            
    simplex = np.dot(theta_t, beta_t)           
            
    X = np.apply_along_axis(lambda x: np.random.normal(x, sd_x), 1, simplex)
    
    simplex_test = np.dot(theta_test, beta_t)
    X_test = np.apply_along_axis(lambda x: np.random.normal(x, sd_x), 1, simplex_test)
            
    return X, beta_t, theta_t, simplex, theta_test, X_test

## Minimum mathcing distance
def min_match(beta, beta_t):
    b_to_t = np.apply_along_axis(lambda x: np.sqrt(((beta_t-x)**2).sum(axis=1)), 1, beta)
    return max([max(np.min(b_to_t, axis=0)), max(np.min(b_to_t, axis=1))])

## NMF objective
def nmf_obj(X, beta, theta='geom'):
  if type(theta)==str:
      theta = np.apply_along_axis(lambda x: proj_on_s(beta, x, beta.shape[0]), 1, X)
      
  est = np.dot(theta, beta)
  
  gauss_ll = norm(X - est)
  
  M = X.shape[0]
  
  return gauss_ll/M
  
def evaluate(X, beta, beta_t, alpha=0.1, theta='geom'):
    
    mm_dist = min_match(beta, beta_t)
    
    if type(theta)==str:
      theta = np.apply_along_axis(lambda x: proj_on_s(beta, x, beta.shape[0]), 1, X)
    
    nmf = nmf_obj(X, beta, theta=theta)
    
    return mm_dist, nmf

## Geometric Theta
def proj_on_s(beta, doc, K, ind_remain=[], first=True, distance=False):
    if first:
        ind_remain = np.arange(K)
    s_0 = beta[0,:]
    if beta.shape[0]==1:
        if distance:
            return norm(doc-s_0)
        else:
            theta = np.zeros(K)
            theta[ind_remain] = 1.
            return theta
    beta_0 = beta[1:,:]
    alpha = lstsq((beta_0-s_0).T, doc-s_0)[0]
    if np.all(alpha>=0) and alpha.sum()<=1:
        if distance:
            p_prime = (alpha*(beta_0-s_0).T).sum(axis=1)
            return norm(doc-s_0-p_prime)
        else:
            theta = np.zeros(K)
            theta[ind_remain] = np.append(1-alpha.sum(), alpha)
            return theta
    elif np.any(alpha<0):
        ind_remain = np.append(ind_remain[0], ind_remain[1:][alpha>0])
        return proj_on_s(np.vstack([s_0, beta_0[alpha>0,:]]), doc, K, ind_remain, False, distance)
    else:
        return proj_on_s(beta_0, doc, K, ind_remain[1:], False, distance)