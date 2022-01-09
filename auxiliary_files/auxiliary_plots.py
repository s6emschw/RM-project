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
    
        plt.xlabel(f"$\lambda$", fontsize = 35)
        plt.ylabel("weights", fontsize = 35)
        plt.axhline(y=0, color='black', linestyle='--')
    
        ax.tick_params(axis='both', which='major', labelsize = 32)

        plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {i[1][0].shape[1]} ", fontsize = 40)
        plt.axis("tight")
    
    plt.savefig(f"{reg_type}_plot_betas.png", dpi=500, bbox_inches='tight')
        
def plot_average_betas(dfs, alphas, reg_type):

    plt.figure(figsize = (40, 25))
    count = 1

    for i, a in zip(dfs, alphas):
        plt.subplot(2, 2, count)
    
        ax = plt.gca()
    
        ax.plot(a, i)
    
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    
        plt.xlabel("alpha", fontsize = 35)
        plt.ylabel("weights", fontsize = 35)
    
        ax.tick_params(axis='both', which='major', labelsize = 32)
     
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axis("tight")
        
        plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {i.shape[1]} ", fontsize = 40)
        
        count += 1
        
def plot_betas_single_case(df, alphas, L_w, reg_type):
    plt.figure(figsize = (20, 10))
    count = 1

    plt.subplot(1, 1, count)

    ax = plt.gca()

    ax.plot(df)

    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis

    plt.xlabel(f"$\lambda$", fontsize = 25)
    plt.ylabel("weights", fontsize = 25)

    ax.tick_params(axis='both', which='major', labelsize = 20)
    ax.set_ylim([0, 6])
    #plt.axhline(y=0, color='black', linestyle='--')
    if reg_type == "Average elastic net":
            
            plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {df.shape[1]}, L_ratio = {L_w}[count-1] ", fontsize = 28)
        
    else:
           
            plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {df.shape[1]} ", fontsize = 28)
    plt.axis("tight")
    count += 1
    
        
def plot_average_betas_elnet(dfs, alphas, L_w, reg_type):

    plt.figure(figsize = (40,30))
    count = 1

    for i, a in zip(dfs, alphas):
        plt.subplot(3, 2, count)
    
        ax = plt.gca()
    
        ax.plot(a, i)
    
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    
        plt.xlabel("alpha", fontsize = 20)
        plt.ylabel("weights", fontsize = 20)
    
        ax.tick_params(axis='both', which='major', labelsize = 20)
        ax.set_ylim([0, 6])
        
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axis("tight")
        
        if reg_type == "Average elastic net":
            plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {i.shape[1]}, L_ratio = {L_w[count-1]} ", fontsize = 28)
        else:
            plt.title(f"{reg_type} coefficients as a function of $\lambda$, p = {i.shape[1]} ", fontsize = 15)
        plt.axis("tight")
        count += 1
        #plt.show()

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
    
    f.savefig('ols_distr.png', dpi=500, bbox_inches='tight')
    
    
def plot_elnet_beta_distribution(df_low, df_med, df_high): 

    sns.set(style="white", palette="muted", color_codes=True)
    #rs = np.random.RandomState(10)

    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 3, figsize=(18, 8))

    fig_1=sns.kdeplot(data=df_low, ax=axes[0])
    fig_1.legend([],[], frameon=False)
    fig_1.title.set_text(f"Distribution of elastic net coefficients for low $\lambda$, p = {df_low.shape[1]}")

    fig_2=sns.kdeplot(data=df_med, ax=axes[1])
    fig_2.legend([],[], frameon=False)
    fig_2.title.set_text(f"Distribution of elastic net coefficients for moderate $\lambda$, p = {df_med.shape[1]}")

    fig_3=sns.kdeplot(data=df_high, ax=axes[2])
    fig_3.legend([],[], frameon=False)
    fig_3.title.set_text(f"Distribution of elastic net coefficients for high $\lambda$, p = {df_high.shape[1]}")
    plt.show()


def plot_shrunken_beta_distribution(df_low, df_med, df_high, reg_type): 

    sns.set(style="white", palette="muted", color_codes=True)

    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 3, figsize=(18, 8))

    fig_1=sns.kdeplot(data=df_low, ax=axes[0])
    fig_1.legend([],[], frameon=False)
    fig_1.axvline(x=5, color='black', linestyle='--')
    fig_1.spines["bottom"].set_linestyle("dotted")
    fig_1.title.set_text(f"Distribution of {reg_type} coefficients for low $\lambda$, p = {df_low.shape[1]}")

    fig_2=sns.kdeplot(data=df_med, ax=axes[1])
    fig_2.legend([],[], frameon=False)
    fig_2.axvline(x=5, color='black', linestyle='--')
    fig_2.spines["bottom"].set_linestyle("dotted")
    fig_2.title.set_text(f"Distribution of {reg_type} coefficients for moderate $\lambda$, p = {df_med.shape[1]}")

    fig_3=sns.kdeplot(data=df_high, ax=axes[2])
    fig_3.legend([],[], frameon=False)
    fig_3.axvline(x=5, color='black', linestyle='--')
    fig_3.spines["bottom"].set_linestyle("dotted")
    fig_3.title.set_text(f"Distribution of {reg_type} coefficients for high $\lambda$, p = {df_high.shape[1]}")
    
    
    
    