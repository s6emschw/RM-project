import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def correlation_matrix(data):
    """
    Plots correlation matrix between variables.
    """
    corr = data.corr()
    plt.figure(figsize=(8,8), dpi=100)
    corr_mat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corr_mat)
    plt.title(f'Correlation Matrix', fontsize=15)
    plt.show()

    
def table_coefficients(coefficients, round_param):
    """
    Saves the coefficients from regression into table.
    """
    names = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
    values = coefficients.round(round_param)
    list_of_tuples = list(zip(names, values))
    table = pd.DataFrame(list_of_tuples, columns = ['Term', 'Coefficient'])
    return table


def plot_model_coefficients(X_train, coefficients, reg_type):
    """
    Plots model coefficients to evaluate the shrinkage.
    """
    df_lm_coefficients = pd.DataFrame(
                            {'predictor': X_train.columns,
                             'coef': coefficients.flatten()}
                      )

    fig, ax = plt.subplots(figsize=(11, 3.84))
    ax.stem(df_lm_coefficients.predictor, df_lm_coefficients.coef, markerfmt=' ')
    plt.xticks(rotation=90, ha='right', size=10)
    ax.set_xlabel('variable')
    ax.set_ylabel('coefficients')
    plt.title(f"{reg_type} model coefficients", fontsize = 15);
    
    
def plot_regression_error(alphas, error, baseline, reg_type, reg_baseline):
    """
    Plots the regression error compared to the error of baseline model.
    """
    plt.figure(figsize=(20,10))
    ax = plt.gca()
    ax.plot(alphas, error, linewidth=2, color='red', label=f'{reg_type} regression')
    ax.plot(alphas, baseline, linewidth=2, linestyle='--', color='blue', label=f'{reg_baseline} regression')
    ax.set_xscale('log')
    plt.xlabel('$\lambda$', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('error', fontsize=20)
    ax.legend(fontsize=20)
    plt.title(r'Regression error ($\lambda$)', fontsize=25)
    plt.show()
    
    
def plot_CV(alphas, val_scores, val_mean):
    """
    Plots MSE on each k-fold cross validation 
    """
    plt.figure(figsize = (20,10))
    ax = plt.gca()
    ax.plot(alphas, val_scores, linewidth=1)
    ax.plot(alphas, val_mean, color="black", linewidth=5, label='Average across the folds')
    ax.axvline(alphas[np.argmin(val_mean)],0,6,linestyle='--',color="black", label='CV selected $\lambda$')
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1]) 
    plt.xlabel("lambda", fontsize = 20)
    plt.ylabel("MSE", fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize = 20)
    plt.axis("tight")
    plt.title("MSE on each 10-fold Cross Validation ", fontsize = 28)
    plt.axis("tight")
    plt.legend()
    
def plot_paths(alphas, val_mean, coefficients, reg_type):
    """
    Plots coefficient paths as a value of alpha/lambda for shrinkage methods.
    """
    plt.figure(figsize=(20,12))
    ax = plt.gca()
    ax.plot(alphas, coefficients)
    ax.set_xscale('log')
    ax.axvline(alphas[np.argmin(val_mean)], 0, 6, linestyle='--', color='black', label='CV selected $\lambda$')
    plt.xlabel('$\lambda$', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel('weight value', fontsize=25)
    plt.title(f'{reg_type} coefficients path', fontsize=30)
    plt.legend()