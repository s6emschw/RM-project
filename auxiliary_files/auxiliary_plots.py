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


def plot_betas(dfs, alphas, reg_type):

    plt.figure(figsize = (40,25))

    for i, a in zip(enumerate(dfs), alphas):
        plt.subplot(2, 2, i[0] + 1)
    
        ax = plt.gca()
    
        ax.plot(a, i[1][0])
    
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    
        plt.xlabel(f"$\lambda$", fontsize = 25)
        plt.ylabel("weights", fontsize = 25)
    
        ax.tick_params(axis='both', which='major', labelsize = 20)

        plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {i[1][0].shape[1]} ", fontsize = 28)
        plt.axis("tight")
        #plt.show()


def plot_average_betas(dfs, alphas, reg_type):

    plt.figure(figsize = (40, 25))
    count = 1

    for i, a in zip(dfs, alphas):
        plt.subplot(2, 2, count)
    
        ax = plt.gca()
    
        ax.plot(a, i)
    
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    
        plt.xlabel(f"$\lambda$", fontsize = 25)
        plt.ylabel("weights", fontsize = 25)
    
        ax.tick_params(axis='both', which='major', labelsize = 20)
        ax.set_ylim([0, 6])

        plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {i.shape[1]} ", fontsize = 28)
        plt.axis("tight")
        count += 1
        #plt.show()
        
def plot_ols_beta_distribution(df): 

    sns.set(style="white", palette="muted", color_codes=True)
    #rs = np.random.RandomState(10)

    # Set up the matplotlib figure
    f, axes = plt.subplots(2, 2, figsize=(20, 20))
    # sns.despine(left=True)

    fig_1=sns.kdeplot(data=df[0], ax=axes[0,0])
    fig_1.legend([],[], frameon=False)
    fig_1.title.set_text('Distribution of OLS coefficients when P=2')

    fig_2=sns.kdeplot(data=df[1], ax=axes[0,1])
    fig_2.legend([],[], frameon=False)
    fig_2.title.set_text('Distribution  of OLS coefficients when P=15')

    fig_3=sns.kdeplot(data=df[2], ax=axes[1,0])
    fig_3.legend([],[], frameon=False)
    fig_3.title.set_text('Distribution of OLS coefficients when P=20')

    fig_4=sns.kdeplot(data=df[3], ax=axes[1,1])
    fig_4.legend([],[], frameon=False)
    fig_4.title.set_text('Distribution coefficients when P=28')
    

def plot_ridge_beta_distribution(df_low, df_med, df_high): 

    sns.set(style="white", palette="muted", color_codes=True)
    #rs = np.random.RandomState(10)

    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 3, figsize=(18, 8))
    # sns.despine(left=True)

    fig_1=sns.kdeplot(data=df_low, ax=axes[0])
    fig_1.legend([],[], frameon=False)
    fig_1.title.set_text(f"Distribution of ridge coefficients for low $\lambda$, p = {df_low.shape[1]}")

    fig_2=sns.kdeplot(data=df_med, ax=axes[1])
    fig_2.legend([],[], frameon=False)
    fig_2.title.set_text(f"Distribution of ridge coefficients for moderate $\lambda$, p = {df_med.shape[1]}")

    fig_3=sns.kdeplot(data=df_high, ax=axes[2])
    fig_3.legend([],[], frameon=False)
    fig_3.title.set_text(f"Distribution of ridge coefficients for high $\lambda$, p = {df_high.shape[1]}")