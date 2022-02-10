import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler
from functools import reduce  
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from mlxtend.evaluate import bias_variance_decomp
from sklearn import metrics
from sklearn import linear_model



def get_sim_data(n, p, cor_factor, true_betas):
    
    sd_vec = np.ones(p) 
    mean = np.zeros(p)
    
    
    cor_matrix = np.zeros([p,p])
    store_corr = []

    for i in list(range(1, p)):
    
        for j in list(range(i + 1, p + 1)): 
            
            corr = cor_factor ** abs(i - j)
            store_corr.append(corr)

    cor_matrix[np.triu_indices(p, 1)] = store_corr
    cor_matrix[np.tril_indices(p, -1)] = cor_matrix.T[np.tril_indices(p, -1)]
    np.fill_diagonal(cor_matrix, 1)
    
    D = np.diag(sd_vec)
    sigma = D.dot(cor_matrix).dot(D)
    
    X = np.random.multivariate_normal(mean, sigma, n)
    
    eps = np.random.normal(0, 1, n)

    y = X.dot(true_betas) + eps 
    
    y = pd.Series(y, name = "y")
    
    column_names = []
    
    for value in range(1, p + 1): 
        
        column = f"X_{value}"
        column_names.append(column)
        
    
    X = pd.DataFrame(X, columns = column_names)
    
    df = pd.concat([y, X], axis = 1)
    
    return y, X, df


def iterate_ridge(n, p, cor_factor, iterations_sim, true_betas, alphas):
    
    beta_var_names = []
    ridge_beta_names = []
    
    for value in range(1, p + 1): 
    
        column_betas_var = f"beta_var_{value}"
        column_betas = f"beta_{value}"
        beta_var_names.append(column_betas_var)
        ridge_beta_names.append(column_betas)

    df_list_betas_ridge = []
    df_list_var_ridge = []

    for i in range(iterations_sim):
        
        true_betas_sim = true_betas
    
        y_noise, X, df = get_sim_data(n, p, cor_factor, true_betas_sim) 
        
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


def iterate_lasso_sklearn(n, p, cor_factor, iterations_sim, true_betas, alphas):
    
    lasso_beta_names = []
    
    for value in range(1, p + 1): 
        column_betas = f"beta_{value}"
        lasso_beta_names.append(column_betas)

    df_list_betas_lasso = []

    for i in range(iterations_sim):
        
        true_betas_sim = true_betas
    
        y_noise, X, df = get_sim_data(n, p, cor_factor, true_betas_sim) 
        matr_beta = []
    
        for a in alphas: 
        
            lasso_model = linear_model.Lasso(alpha=a).fit(X,y_noise) 
            lasso_beta = np.array(lasso_model.coef_)
            matr_beta.append(lasso_beta)
            df_lasso_betas = pd.DataFrame(matr_beta, columns = lasso_beta_names)
        
         
        df_list_betas_lasso.append(df_lasso_betas)
        
    return df_list_betas_lasso


def iterate_elnet(n, p, cor_factor, iterations_sim, true_betas, alphas, L_w):
    
    elnet_beta_names = []
    
    for value in range(1, p + 1): 
    
        column_betas = f"beta_{value}"
        elnet_beta_names.append(column_betas)

    df_list_betas_elnet = []

    for i in range(iterations_sim):
        
       # true_betas_sim = true_betas
    
        y, X, df = get_sim_data(n, p, cor_factor, true_betas) 
        matr_beta = []
        
        #for L_w in L_weight:
    
        for a in alphas: 
        
            elnet_model = ElasticNet(alpha=a, l1_ratio=L_w).fit(X,y)
            elnet_beta = np.array(elnet_model.coef_)
            matr_beta.append(elnet_beta)
            df_elnet_betas = pd.DataFrame(matr_beta, columns = elnet_beta_names)
        
         
        df_list_betas_elnet.append(df_elnet_betas)
        
    return df_list_betas_elnet


def get_var_distribution(df, iterations, alpha_low, alpha_med, alpha_high):

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


def generate_true_betas(non_zero_betas, zero_betas, size_nonzero):

    store_true_betas = []

    for i, j in zip(non_zero_betas, zero_betas): 
    
        non_zeros = np.repeat(size_nonzero, i)
        zeros = np.repeat(0, j)
    
        true_betas = np.concatenate([non_zeros, zeros])
        store_true_betas.append(true_betas)
        
    return store_true_betas


def get_predictions(n, p, true_betas, cor_factor, iterations, alphas, X_test): 

    store_predictions_list_ridge = []
    store_predictions_list_lasso = []
    store_predictions_list_elnet_20 = []
    store_predictions_list_elnet_50 = []
    store_predictions_list_elnet_70 = []

    for i in range(iterations):
    
        store_predictions_ridge = []
        store_predictions_lasso = []
        store_predictions_elnet_20 = []
        store_predictions_elnet_50 = []
        store_predictions_elnet_70 = []
    
        y_train, X_train, df_train = get_sim_data(n, p, cor_factor, true_betas) # get test data 

        for a in alphas: 

            ridge = Ridge(alpha=a).fit(X_train, y_train)
            ridge_predict = ridge.predict(X_test)
            ridge_predict_select = ridge_predict[14]
            store_predictions_ridge.append(ridge_predict_select) 
        
            lasso = Lasso(alpha=a).fit(X_train, y_train)
            lasso_predict = lasso.predict(X_test)
            lasso_predict_select = lasso_predict[14]
            store_predictions_lasso.append(lasso_predict_select) 
        
            elnet_20 = ElasticNet(alpha=a, l1_ratio=0.2).fit(X_train, y_train)
            elnet_20_predict = elnet_20.predict(X_test)
            elnet_20_predict_select = elnet_20_predict[14]
            store_predictions_elnet_20.append(elnet_20_predict_select)
        
            elnet_50 = ElasticNet(alpha=a, l1_ratio=0.5).fit(X_train, y_train)
            elnet_50_predict = elnet_50.predict(X_test)
            elnet_50_predict_select = elnet_50_predict[14]
            store_predictions_elnet_50.append(elnet_50_predict_select)
        
            elnet_70 = ElasticNet(alpha=a, l1_ratio=0.7).fit(X_train, y_train)
            elnet_70_predict = elnet_70.predict(X_test)
            elnet_70_predict_select = elnet_70_predict[14]
            store_predictions_elnet_70.append(elnet_70_predict_select)
    
        store_predictions_list_ridge.append(store_predictions_ridge)
        store_predictions_list_lasso.append(store_predictions_lasso)
        store_predictions_list_elnet_20.append(store_predictions_elnet_20)
        store_predictions_list_elnet_50.append(store_predictions_elnet_50)
        store_predictions_list_elnet_70.append(store_predictions_elnet_70)
        
        store_predictions_df_ridge = pd.DataFrame(store_predictions_list_ridge)
        store_predictions_df_lasso = pd.DataFrame(store_predictions_list_lasso)
        store_predictions_df_elnet_20 = pd.DataFrame(store_predictions_list_elnet_20)
        store_predictions_df_elnet_50 = pd.DataFrame(store_predictions_list_elnet_50)
        store_predictions_df_elnet_70 = pd.DataFrame(store_predictions_list_elnet_70)
        
    predictions_dfs = [store_predictions_df_ridge, store_predictions_df_lasso, store_predictions_df_elnet_20,
                           store_predictions_df_elnet_50, store_predictions_df_elnet_70]
        
    return predictions_dfs


def compute_mse(predictions_df_list, y_test, iterations):

    store_mse_lists = []
    store_variance_lists = []
    store_bias_sq_lists = []

    for df in enumerate(predictions_df_list):
    
        store_mse = []
        store_variance = []
        store_bias_sq = []
    
        for i in df[1].columns: 

            mse = np.sum((np.asarray(df[1].iloc[:,i]) - y_test.iloc[14])**2) / iterations
            variance = np.mean((np.mean(df[1].iloc[:,i]) - np.asarray(df[1].iloc[:,i]))**2)
            bias_squared = (np.mean(df[1].iloc[:,i]) - y_test.iloc[14])**2
    
            store_mse.append(mse)
            store_variance.append(variance)
            store_bias_sq.append(bias_squared)
    
        store_mse_lists.append(store_mse)
        store_variance_lists.append(store_variance)
        store_bias_sq_lists.append(store_bias_sq)
    
    return store_mse_lists, store_variance_lists, store_bias_sq_lists


