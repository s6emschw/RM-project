import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from functools import reduce  
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from mlxtend.evaluate import bias_variance_decomp
from sklearn import metrics


def plot_betas(dfs, alphas, rows, columns, reg_type):

    plt.figure(figsize = (40,25))

    for i, a in zip(enumerate(dfs), alphas):
        plt.subplot(rows, columns, i[0] + 1)
    
        ax = plt.gca()
    
        ax.plot(a, i[1][0])
    
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    
        plt.xlabel(f"$\lambda$", fontsize = 35)
        plt.ylabel("weights", fontsize = 35)
        plt.axhline(y=0, color='black', linestyle='--')
    
        ax.tick_params(axis='both', which='major', labelsize = 32)

        plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {i[1][0].shape[1]} ", fontsize = 40)
        plt.axis("tight")
    
    plt.savefig(f"{reg_type}_plot_betas.png", bbox_inches='tight')

def plot_betas_lasso(dfs, alphas, rows, columns, sparsity_index):

    plt.figure(figsize = (30,30))
    count = 1

    for i, a, s in zip(dfs, alphas, sparsity_index):
        plt.subplot(rows, columns, count)
    
        ax = plt.gca()
    
        ax.plot(a, i)
    
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    
        plt.xlabel(f"$\lambda$", fontsize = 35)
        plt.ylabel("weights", fontsize = 35)
        plt.axhline(y=0, color='black', linestyle='--')
    
        ax.tick_params(axis='both', which='major', labelsize = 32)
        ax.set_ylim([0, 6])

        plt.title(f"Lasso coefficients as a function of $\lambda$, s$_{{{0}}}$ = {s}", fontsize = 40)
        plt.axis("tight")
        
        count += 1
    
    plt.savefig("lasso_plot_betas.png", bbox_inches='tight')
    
        
def plot_average_ridge_betas(dfs, alphas, reg_type):

    plt.figure(figsize = (40, 25))
    count = 1

    for i, a in zip(dfs, alphas):
        plt.subplot(2, 2, count)
    
        ax = plt.gca()
    
        ax.plot(a, i)
    
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    
        plt.xlabel(f"$\lambda$", fontsize = 35)
        plt.ylabel("weights", fontsize = 35)
    
        ax.tick_params(axis='both', which='major', labelsize = 32)
     
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axis("tight")
        
        plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {i.shape[1]} ", fontsize = 40)
        
        count += 1
    
    plt.savefig(f"{reg_type}_plot_average_betas.png", bbox_inches='tight')


def plot_average_lasso_betas(dfs, alphas, sparsity_index, reg_type):

    plt.figure(figsize = (40, 25))
    count = 1

    for i, a, s in zip(dfs, alphas, sparsity_index):
        plt.subplot(2, 2, count)
    
        ax = plt.gca()
    
        ax.plot(a, i)
    
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    
        plt.xlabel(f"$\lambda$", fontsize = 35)
        plt.ylabel("weights", fontsize = 35)
    
        ax.tick_params(axis='both', which='major', labelsize = 32)
     
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axis("tight")
        
        plt.title(f"{reg_type} coefficients as a function of $\lambda$, s$_{{{0}}}$ = {s} ", fontsize = 40)
        
        count += 1
    
    plt.savefig(f"{reg_type}_plot_average_betas.png", bbox_inches='tight')
       

def plot_average_betas_elnet(dfs, alphas, L_w):

    plt.figure(figsize = (50,40))
    count = 1

    for i, a in zip(dfs, alphas):
        plt.subplot(2, 3, count)
    
        ax = plt.gca()
    
        ax.plot(a, i)
    
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    
        plt.xlabel(f"$\lambda$", fontsize = 35)
        plt.ylabel("weights", fontsize = 35)
    
        ax.tick_params(axis='both', which='major', labelsize = 32)
        ax.set_ylim([0, 6])
        
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axis("tight")
        
        plt.title(f"Average elastic net coefficients, L_ratio = {L_w[count-1]} ", fontsize = 40)
      
        plt.axis("tight")
        count += 1
    
    plt.savefig("elastic_net_plot_average_betas.png", bbox_inches='tight')

def plot_ols_beta_distribution(df): 

    sns.set(style="white", palette="muted", color_codes=True)

    # Set up the matplotlib figure
    f, axes = plt.subplots(2, 2, figsize=(20, 20))

    fig_1=sns.kdeplot(data=df[0], ax=axes[0,0])
    fig_1.legend([],[], frameon=False)
    fig_1.axvline(x=5, color='black', linestyle='--')
    fig_1.spines["bottom"].set_linestyle("dotted")
    fig_1.tick_params(labelsize=25)
    fig_1.set_ylabel("Density", fontsize=28)
    fig_1.set_title(f"Distribution of OLS coefficients for p = {df[0].shape[1]}", fontsize=32)
    
    fig_2=sns.kdeplot(data=df[1], ax=axes[0,1])
    fig_2.legend([],[], frameon=False)
    fig_2.axvline(x=5, color='black', linestyle='--')
    fig_2.spines["bottom"].set_linestyle("dotted")
    fig_2.tick_params(labelsize=25)
    fig_2.set_ylabel("Density", fontsize=28)
    fig_2.set_title(f"Distribution  of OLS coefficients for p = {df[1].shape[1]}", fontsize=32)

    fig_3=sns.kdeplot(data=df[2], ax=axes[1,0])
    fig_3.legend([],[], frameon=False)
    fig_3.axvline(x=5, color='black', linestyle='--')
    fig_3.spines["bottom"].set_linestyle("dotted")
    fig_3.tick_params(labelsize=25)
    fig_3.set_ylabel("Density", fontsize=28)
    fig_3.set_title(f"Distribution of OLS coefficients for p = {df[2].shape[1]}", fontsize=32)

    fig_4=sns.kdeplot(data=df[3], ax=axes[1,1])
    fig_4.legend([],[], frameon=False)
    fig_4.axvline(x=5, color='black', linestyle='--')
    fig_4.spines["bottom"].set_linestyle("dotted")
    fig_4.tick_params(labelsize=25)
    fig_4.set_ylabel("Density", fontsize=28)
    fig_4.set_title(f"Distribution coefficients for p = {df[3].shape[1]}", fontsize=32)
    
    f.savefig('ols_distr.png', bbox_inches='tight')
    

def plot_shrunken_beta_distribution(df_low, df_med, df_high, reg_type, nonzero_betas_mean, L_weight): 

    sns.set(style="white", palette="muted", color_codes=True)

    # Set up the matplotlib figure
    
    
    if reg_type == "elastic net": 
        
        f, axes = plt.subplots(1, 3, figsize=(80, 30))
        
        fig_1=sns.kdeplot(data=df_low, ax=axes[0])
        fig_1.legend([],[], frameon=False)
        fig_1.axvline(x=nonzero_betas_mean, color='blue', linestyle='--')
        fig_1.spines["bottom"].set_linestyle("dotted")
        fig_1.tick_params(labelsize=40)
        fig_1.set_ylabel("Density", fontsize=45)
        fig_1.set_title(f"Distribution of {reg_type} coefficients, low $\lambda$, L_ratio = {L_weight}", fontsize=50)

        fig_2=sns.kdeplot(data=df_med, ax=axes[1])
        fig_2.legend([],[], frameon=False)
        fig_2.axvline(x=nonzero_betas_mean, color='blue', linestyle='--')
        fig_2.spines["bottom"].set_linestyle("dotted")
        fig_2.tick_params(labelsize=40)
        fig_2.set_ylabel("Density", fontsize=45)
        fig_2.set_title(f"Distribution of {reg_type} coefficients, moderate $\lambda$, L_ratio = {L_weight}", fontsize=50)

        fig_3=sns.kdeplot(data=df_high, ax=axes[2])
        fig_3.legend([],[], frameon=False)
        fig_3.axvline(x=nonzero_betas_mean, color='blue', linestyle='--')
        fig_3.spines["bottom"].set_linestyle("dotted")
        fig_3.tick_params(labelsize=40)
        fig_3.set_ylabel("Density", fontsize=45)
        fig_3.set_title(f"Distribution of {reg_type} coefficients, high $\lambda$, L_ratio = {L_weight}", fontsize=50)
    
        f.savefig(f"{reg_type}_shrunken_beta_dist_{df_low.shape[1]}_{L_weight}.png", bbox_inches='tight')
    
    else: 
        
        f, axes = plt.subplots(1, 3, figsize=(50, 16))

        fig_1=sns.kdeplot(data=df_low, ax=axes[0])
        fig_1.legend([],[], frameon=False)
        fig_1.axvline(x=nonzero_betas_mean, color='blue', linestyle='--')
        fig_1.spines["bottom"].set_linestyle("dotted")
        fig_1.tick_params(labelsize=32)
        fig_1.set_ylabel("Density", fontsize=37)
        fig_1.set_title(f"Distribution of {reg_type} coefficients, low $\lambda$", fontsize=40)

        fig_2=sns.kdeplot(data=df_med, ax=axes[1])
        fig_2.legend([],[], frameon=False)
        fig_2.axvline(x=nonzero_betas_mean, color='blue', linestyle='--')
        fig_2.spines["bottom"].set_linestyle("dotted")
        fig_2.tick_params(labelsize=32)
        fig_2.set_ylabel("Density", fontsize=37)
        fig_2.set_title(f"Distribution of {reg_type} coefficients, moderate $\lambda$", fontsize=40)

        fig_3=sns.kdeplot(data=df_high, ax=axes[2])
        fig_3.legend([],[], frameon=False)
        fig_3.axvline(x=nonzero_betas_mean, color='blue', linestyle='--')
        fig_3.spines["bottom"].set_linestyle("dotted")
        fig_3.tick_params(labelsize=32)
        fig_3.set_ylabel("Density", fontsize=37)
        fig_3.set_title(f"Distribution of {reg_type} coefficients, high $\lambda$", fontsize=40)
    
        f.savefig(f"{reg_type}_shrunken_beta_dist_{df_low.shape[1]}.png", bbox_inches='tight')


def bias_var_tradeoff(alphas, mse, var, bias_sq):

    plt.figure(figsize = (10,10))

    ax = plt.gca()

    mse = ax.plot(alphas, mse, label = "MSE")
    variance = ax.plot(alphas, var, label = "variance")
    bias_sq = ax.plot(alphas, bias_sq, label = "squared bias")
    ax.set_xscale("log")
    plt.xlabel(f"$\lambda$", fontsize = 20)
    plt.ylabel("MSE", fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize = 20)
    plt.axis("tight")
    plt.title("Bias-Variance Tradeoff", fontsize = 28)
    ax.legend(["MSE", "variance", "squared bias"], fontsize=15)

    plt.savefig("bias_var_tradeoff.png", bbox_inches='tight')



def plot_cv_sim():

    plt.figure(figsize = (25,10))

    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.plot(alphas, val_mean_ridge, color="red", label = "10-fold CV") # CV average ridge 10-fold
    ax.plot(alphas, val_mean_ridge_loo, color="blue", label = "leave one out CV") # CV average ridge leave one out
    #ax.plot(alphas, val_mean_lasso, color="purple") # CV average lasso 10-fold
    #ax.plot(alphas, val_mean_lasso_loo, color="pink") # CV average lasso leave one out 
    ax.plot(alphas, store_mse_1[0], color="black", label = "estimation of true mse") # true ridge
    #ax.plot(alphas, store_mse_1[1], color="blue") # true lasso 
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1]) 
    minimum_10_cv_ridge = [alphas[np.argmin(val_mean_ridge)], np.min(val_mean_ridge)]
    minimum_loo_ridge = [alphas[np.argmin(val_mean_ridge_loo)], np.min(val_mean_ridge_loo)]
    minimum_mse_true_ridge = [alphas[np.argmin(store_mse_1[0])], np.min(store_mse_1[0])]
    ax.plot(*minimum_10_cv_ridge, "X", color="red")
    ax.plot(*minimum_loo_ridge, "X", color="blue")
    ax.plot(*minimum_mse_true_ridge, "X", color="black")
    plt.xlabel(f"$\lambda$", fontsize = 20)
    plt.ylabel("MSE", fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize = 20)
    ax.legend(fontsize=15)
    plt.axis("tight")
    plt.title("Validation curve, ridge", fontsize = 28)
    plt.axis("tight")

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    #ax.plot(alphas, val_mean_ridge, color="dimgrey") # CV average ridge 10-fold
    #ax.plot(alphas, val_mean_ridge_loo, color="green") # CV average ridge leave one out
    ax.plot(alphas, val_mean_lasso, color="red", label = "10-fold CV") # CV average lasso 10-fold
    ax.plot(alphas, val_mean_lasso_loo, color="blue", label = "leave one out CV") # CV average lasso leave one out 
    #ax.plot(alphas, store_mse_1[0], color="red") # true ridge
    ax.plot(alphas, store_mse_1[1], color="black", label = "estimation of true mse") # true lasso 
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])
    minimum_10_cv_lasso = [alphas[np.argmin(val_mean_lasso)], np.min(val_mean_lasso)]
    minimum_loo_lasso = [alphas[np.argmin(val_mean_lasso_loo)], np.min(val_mean_lasso_loo)]
    minimum_mse_true_lasso = [alphas[np.argmin(store_mse_1[1])], np.min(store_mse_1[1])]
    ax.plot(*minimum_10_cv_lasso, "X", color="red")
    ax.plot(*minimum_loo_lasso, "X", color="blue")
    ax.plot(*minimum_mse_true_lasso, "X", color="black")
    plt.xlabel(f"$\lambda$", fontsize = 20)
    plt.ylabel("MSE", fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize = 20)
    ax.legend(fontsize=15)
    plt.axis("tight")
    plt.title("Validation curve, lasso", fontsize = 28)
    plt.axis("tight") 

    plt.savefig("CV_simulation.png", bbox_inches='tight')


    