import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from functools import reduce  
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from mlxtend.evaluate import bias_variance_decomp
from sklearn import metrics
from sklearn import linear_model



def get_sim_data(n, p, q, min_cor, max_cor, true_betas):
    
    #p is the number of correlated regressors
    #q is the number of uncorrelated regressors
    #p+q= tot number of regressros
    
    sd_vec = np.ones(p) 
    mean = np.zeros(p)
    cor_matrix = np.zeros((p,p))

    correlation = np.random.uniform(min_cor, max_cor, int(p * (p - 1) / 2))
    cor_matrix[np.triu_indices(p, 1)] = correlation
    cor_matrix[np.tril_indices(p, -1)] = cor_matrix.T[np.tril_indices(p, -1)]
    np.fill_diagonal(cor_matrix, 1)


    D = np.diag(sd_vec)
    sigma = D.dot(cor_matrix).dot(D)

    X_corr = np.random.multivariate_normal(mean, sigma, n)
    if q>0:
        X_uncorr = np.random.multivariate_normal(np.zeros(q), np.identity(q), n)
        X = np.concatenate([X_corr, X_uncorr], axis=1) #X = pd.concat([X_corr, X_uncorr], axis=1)
    else:
        X = X_corr
        
    eps = np.random.normal(0, 1, n)

    y = X.dot(true_betas) + eps 
    
    y = pd.Series(y, name = "y")
    
    column_names = []
    
    for value in range(1, p+q + 1): 
        
        column = f"X_{value}"
        column_names.append(column)
        
    
    X = pd.DataFrame(X, columns = column_names)
    
    df = pd.concat([y, X], axis = 1)
    
    return y, X, df


def iterate_ridge(n, p, q, min_cor, max_cor, iterations_sim, true_betas, alphas):
    
    beta_var_names = []
    ridge_beta_names = []
    
    for value in range(1, p + q + 1): 
    
        column_betas_var = f"beta_var_{value}"
        column_betas = f"beta_{value}"
        beta_var_names.append(column_betas_var)
        ridge_beta_names.append(column_betas)

    df_list_betas_ridge = []
    df_list_var_ridge = []

    for i in range(iterations_sim):
        
        true_betas_sim = true_betas
    
        y_noise, X, df = get_sim_data(n, p, q, min_cor, max_cor, true_betas_sim) 
        
        i, k = X.shape
        I = np.identity(k)
    
        matr_var = []
        matr_beta = []
    

        for a in alphas: 
        
            ridge_beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + a * I), X.T), y_noise)
            matr_beta.append(ridge_beta)
            df_ridge_betas = pd.DataFrame(matr_beta, columns = ridge_beta_names)
        
            ridge_var_cov = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + a * I), np.dot(X.T, X)), np.linalg.inv(np.dot(X.T, X) + a * I))
            ridge_var = ridge_var_cov.diagonal()
            matr_var.append(ridge_var)
            df_ridge_var = pd.DataFrame(matr_var, columns = beta_var_names)
        
         
        df_list_betas_ridge.append(df_ridge_betas)
        df_list_var_ridge.append(df_ridge_var)
        
    return df_list_betas_ridge, df_list_var_ridge


def iterate_lasso_sklearn(n, p, q, min_cor, max_cor, iterations_sim, true_betas, alphas):
    
    lasso_beta_names = []
    
    for value in range(1, p + 1): 
        column_betas = f"beta_{value}"
        lasso_beta_names.append(column_betas)

    df_list_betas_lasso = []

    for i in range(iterations_sim):
        
        true_betas_sim = true_betas
    
        y_noise, X, df = get_sim_data(n, p, q, min_cor, max_cor, true_betas_sim) 
        matr_beta = []
    
        for a in alphas: 
        
            lasso_model = linear_model.Lasso(alpha=a, fit_intercept=False).fit(X,y_noise) 
            lasso_beta = np.array(lasso_model.coef_)
            matr_beta.append(lasso_beta)
            df_lasso_betas = pd.DataFrame(matr_beta, columns = lasso_beta_names)
        
         
        df_list_betas_lasso.append(df_lasso_betas)
        
    return df_list_betas_lasso

def iterate_elnet(n, p, q, min_cor, max_cor, iterations_sim, true_betas, alphas, L_w):
    
    elnet_beta_names = []
    
    for value in range(1, p+q + 1): 
    
        column_betas = f"beta_{value}"
        elnet_beta_names.append(column_betas)

    df_list_betas_elnet = []

    for i in range(iterations_sim):
        
        true_betas_sim = true_betas
    
        y, X, df = get_sim_data(n, p, q, min_cor, max_cor, true_betas_sim) 
        matr_beta = []
        
        #for L_w in L_weight:
    
        for a in alphas: 
        
            elnet_model = ElasticNet(alpha=a, l1_ratio=L_w, fit_intercept=False).fit(X,y)
            elnet_beta = np.array(elnet_model.coef_)
            matr_beta.append(elnet_beta)
            df_elnet_betas = pd.DataFrame(matr_beta, columns = elnet_beta_names)
        
         
        df_list_betas_elnet.append(df_elnet_betas)
        
    return df_list_betas_elnet

def get_ridge_var_distribution(df, iterations, alpha_low, alpha_med, alpha_high):


    betas_low_alpha = []
    betas_med_alpha = []
    betas_high_alpha = []

    for i in list(range(iterations)): 

        betas_low_a = np.array(df[i].iloc[alpha_low, :])
        betas_med_a = np.array(df[i].iloc[alpha_med, :])
        betas_high_a = np.array(df[i].iloc[alpha_high, :])
    
        betas_low_alpha.append(betas_low_a)
        betas_med_alpha.append(betas_med_a)
        betas_high_alpha.append(betas_high_a)
    
        df_low_a = pd.DataFrame(betas_low_alpha)
        df_med_a = pd.DataFrame(betas_med_alpha)
        df_high_a = pd.DataFrame(betas_high_alpha)
        
    
    return df_low_a, df_med_a, df_high_a

def get_lasso_var_distribution(df, iterations, alpha_low, alpha_med, alpha_high):


    betas_low_alpha = []
    betas_med_alpha = []
    betas_high_alpha = []

    for i in list(range(iterations)): 

        betas_low_a = np.array(df[i].iloc[alpha_low, :])
        betas_med_a = np.array(df[i].iloc[alpha_med, :])
        betas_high_a = np.array(df[i].iloc[alpha_high, :])
    
        betas_low_alpha.append(betas_low_a)
        betas_med_alpha.append(betas_med_a)
        betas_high_alpha.append(betas_high_a)
    
        df_low_a = pd.DataFrame(betas_low_alpha)
        df_med_a = pd.DataFrame(betas_med_alpha)
        df_high_a = pd.DataFrame(betas_high_alpha)
        
    
    return df_low_a, df_med_a, df_high_a

def get_elnet_var_distribution(df, iterations, alpha_low, alpha_med, alpha_high):


    betas_low_alpha = []
    betas_med_alpha = []
    betas_high_alpha = []

    for i in list(range(iterations)): 

        betas_low_a = np.array(df[i].iloc[alpha_low, :])
        betas_med_a = np.array(df[i].iloc[alpha_med, :])
        betas_high_a = np.array(df[i].iloc[alpha_high, :])
    
        betas_low_alpha.append(betas_low_a)
        betas_med_alpha.append(betas_med_a)
        betas_high_alpha.append(betas_high_a)
    
        df_low_a = pd.DataFrame(betas_low_alpha)
        df_med_a = pd.DataFrame(betas_med_alpha)
        df_high_a = pd.DataFrame(betas_high_alpha)
        
    
    return df_low_a, df_med_a, df_high_a



def generate_true_betas(non_zero_betas, zero_betas):

    store_true_betas = []

    for i, j in zip(non_zero_betas, zero_betas): 
    
        non_zeros = np.repeat(5, i)
        zeros = np.repeat(0, j)
    
        true_betas = np.concatenate([non_zeros, zeros])
        store_true_betas.append(true_betas)
        
    return store_true_betas


def gen_true_betas_lasso(non_zero_betas, zero_betas):

    true_betas = []

    for i, j in zip(non_zero_betas, zero_betas): 
    
        non_zeros = np.repeat(5, i)
        zeros = np.repeat(0, j)
    
        true_betas = np.concatenate([non_zeros, zeros])
        
    return true_betas