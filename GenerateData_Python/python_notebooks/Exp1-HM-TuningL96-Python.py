# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python [conda env:envHM] *
#     language: python
#     name: conda-env-envHM-py
# ---

# %% [markdown]
# # History Matching L96
# Author: Redouane Lguensat
#
# Date: 26/07/2021

# %%
from L96 import * #https://github.com/raspstephan/Lorenz-Online
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
np.random.seed(42)

X_init = 10 * np.ones(36)
X_init[18] = 10 + 0.01

l96_two = L96TwoLevelOriginal(K=36, save_dt=0.001, X_init=X_init, integration_type='coupled') # True params J=10, h=1, F=10, c=10, b=10

# %%
l96_two.iterate(10)

# %%
l96_two.history.X.plot()

# %% [markdown]
# #### take the last state as an init for a new simulation
# since now we are sure that it starts from the attractor

# %%
Newinit = l96_two.history.X[-1,:].values
Newinit

# %%
np.random.seed(42)

l96_two_truth = L96TwoLevelOriginal(K=36, save_dt=0.001, X_init=Newinit, integration_type='coupled') 

# %%
l96_two_truth.iterate(100)

# %% [markdown]
# This is our ground truth

# %%
l96_two_truth.history.X.plot()

# %% [markdown]
# # Metrics

# %% [markdown]
# We will use the 5 terms depicted here as in Schneider et al. 2017. 
#
# $$
# \boldsymbol{f}(X, Y)=\left(\begin{array}{c}
# X \\
# \bar{Y} \\
# X^{2} \\
# X \bar{Y}\\
# \bar{Y}^{2}
# \end{array}\right)
# $$

# %% [markdown]
# # History Matching

# %% [markdown]
# We want to find the parameters (h, F, c, b)
#
# $\frac{d X_{k}}{d t}=\underbrace{-X_{k-1}\left(X_{k-2}-X_{k+1}\right)}_{\text {Advection}} \underbrace{-X_{k}}_{\text {Diffusion}}+\underbrace{F}_{\text {Forcing}} - \underbrace{\frac{h c}{b} \Sigma_j Y_{j,k}}_{\text {Coupling}}$
#
# $ \frac{d Y_{j, k}}{d t}=\underbrace{-b c Y_{j+1, k}\left(Y_{j+2, k}-Y_{j-1, k}\right)}_{\text {Advection }} \underbrace{- c Y_{j, k}}_{\text {Diffusion }} \underbrace{+\frac{h c}{b} X_{k}}_{\text {Coupling }}$

# %% [markdown]
# # Wave1

# %% [markdown]
# calculate metrics for the true trajectory

# %%
TrueTraj = l96_two_truth.mean_stats(ax=0)


# %% [markdown]
# Define metric 

# %%
def metric(params):
    l96param_spinup = L96TwoLevelOriginal(K=36,
                           save_dt=0.001,
                           X_init=X_init,
                           h=params[0],
                           F=params[1],
                           c=params[2],
                           b=params[3],
                           integration_type='coupled') 
    l96param_spinup.iterate(10)
    l96param = L96TwoLevelOriginal(K=36,
                           save_dt=0.001,
                           X_init=l96param_spinup.history.X[-1,:].values,
                           h=params[0],
                           F=params[1],
                           c=params[2],
                           b=params[3],
                           integration_type='coupled') 
    l96param.iterate(100)
    return l96param.mean_stats(ax=0)


# %% [markdown]
# Sanity check with the true params

# %%
sancheck = metric(np.array([1,10,10,10]))

# %%
np.mean((TrueTraj - sancheck)**2)

# %% [markdown]
# # Running the 40 simulations in parallel

# %%
inputs = pd.read_csv("../Data/df_inputs_newPCA_wave1.csv").values

# %%
inputs

# %%
plt.figure(figsize=(15,15))

plt.subplot(2,3,2)
plt.plot(inputs[:,1], inputs[:,2], ".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("c")
plt.legend()


plt.subplot(2,3,1)
plt.plot(inputs[:,0], inputs[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("F")
plt.legend()

plt.subplot(2,3,3)
plt.plot(inputs[:,2], inputs[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

plt.subplot(2,3,4)
plt.plot(inputs[:,3], inputs[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("b")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,5)
plt.plot(inputs[:,2], inputs[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("c")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,6)
plt.plot(inputs[:,1], inputs[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("b")
plt.legend()

# %%
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
num_cores

# %%
import joblib
joblib.__version__

# %%
# %%time

#feel free to use a higher n_jobs if you can afford to
results = Parallel(n_jobs=20)(delayed(metric)(i) for i in inputs) 

targets = np.array(results)

# %%
inputs = inputs[np.all(np.isfinite(targets), axis=1)]
targets = targets[np.all(np.isfinite(targets), axis=1)]

# %%
targets.shape, inputs.shape

# %% [markdown]
# # pca

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def reduceL96pca(vector):   
    scaler = StandardScaler().fit(vector)
    datascaled = scaler.transform(vector)
    TrueTrajscaled = scaler.transform(TrueTraj[None,:])
    ###########
    pca = PCA(n_components=0.99, svd_solver = 'full')
    pca.fit(datascaled)
    reduceddata = pca.transform(datascaled)
    reducedTrueTraj = pca.transform(TrueTrajscaled)
    return reduceddata, reducedTrueTraj, scaler, pca


# %%
targetsreduced, Obsreduced, sc, pc = reduceL96pca(targets)
targetsreduced.shape, Obsreduced.shape

# %%
sns.set_context("talk")

sns.scatterplot(x=targetsreduced[:,0],y=targetsreduced[:,1],
                hue=targetsreduced[:,0], palette='coolwarm', legend=False)
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*', label='Ground Truth')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# %% [markdown]
# ### save dataframes

# %%
df_metrics = pd.DataFrame()
for i in range(targetsreduced.shape[1]):
    df_metrics['pca_'+str(i)] = targetsreduced[:,i]
df_metrics.to_csv('../Data/df_metrics_newPCA_wave1.csv', index=False)

# %%
df_metrics

# %%
df_obs_nonoise = pd.DataFrame()
for i in range(targetsreduced.shape[1]):
    df_obs_nonoise['pca_'+str(i)] = Obsreduced[:,i]
df_obs_nonoise.to_csv('../Data/df_obs_nonoise_newPCA_wave1.csv', index=False)

# %% [markdown]
# ## 2) wave 2

# %%
inputs2 = pd.read_csv("../Data/exp_TuningL94_newPCA_wave2.csv").values

# %%
sns.set_context("talk")

plt.figure(figsize=(15,15))

plt.subplot(2,3,2)
plt.plot(inputs2[:,1], inputs2[:,2], ".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("c")
plt.legend()


plt.subplot(2,3,1)
plt.plot(inputs2[:,0], inputs2[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("F")
plt.legend()

plt.subplot(2,3,3)
plt.plot(inputs2[:,2], inputs2[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

plt.subplot(2,3,4)
plt.plot(inputs2[:,3], inputs2[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("b")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,5)
plt.plot(inputs2[:,2], inputs2[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("c")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,6)
plt.plot(inputs2[:,1], inputs2[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("b")
plt.legend()

# %%
inputs2.shape

# %%
# %%time

results2 = Parallel(n_jobs=20, verbose=1)(delayed(metric)(i) for i in inputs2)

targets2 = np.array(results2)

# %%
inputs2 = inputs2[np.all(np.isfinite(targets2), axis=1)]
targets2 = targets2[np.all(np.isfinite(targets2), axis=1)]


# %%
def reduceL96pca_usingwave1(vector, scaler, pca): 
    datascaled = scaler.transform(vector)
    return pca.transform(datascaled)


# %%
targetsreduced2 = reduceL96pca_usingwave1(targets2, sc, pc)
targetsreduced2.shape

# %%
sns.set_context("talk")

sns.scatterplot(x=targetsreduced2[:,0],y=targetsreduced2[:,1],
                hue=targetsreduced2[:,0], palette='coolwarm', legend=False)
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*', label='Ground Truth')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# %%
df_metrics = pd.DataFrame()
for i in range(targetsreduced2.shape[1]):
    df_metrics['pca_'+str(i)] = targetsreduced2[:,i]
df_metrics.to_csv('../Data/df_metrics_newPCA_wave2.csv', index=False)

# %% [markdown]
# # wave3

# %%
inputs3 = pd.read_csv("../Data/exp_TuningL94_newPCA_wave3.csv").values

# %%
sns.set_context("talk")

plt.figure(figsize=(15,15))

plt.subplot(2,3,2)
plt.plot(inputs3[:,1], inputs3[:,2], ".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("c")
plt.legend()


plt.subplot(2,3,1)
plt.plot(inputs3[:,0], inputs3[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("F")
plt.legend()

plt.subplot(2,3,3)
plt.plot(inputs3[:,2], inputs3[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

plt.subplot(2,3,4)
plt.plot(inputs3[:,3], inputs3[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("b")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,5)
plt.plot(inputs3[:,2], inputs3[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("c")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,6)
plt.plot(inputs3[:,1], inputs3[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("b")
plt.legend()

# %%
inputs3.shape

# %%
# %%time

results3 = Parallel(n_jobs=20)(delayed(metric)(i) for i in inputs3)

targets3 = np.array(results3)

# %%
inputs3.shape, targets3.shape

# %%
inputs3 = inputs3[np.all(np.isfinite(targets3), axis=1)]
targets3 = targets3[np.all(np.isfinite(targets3), axis=1)]

# %%
inputs3.shape, targets3.shape

# %%
targetsreduced3 = reduceL96pca_usingwave1(targets3, sc, pc)
targetsreduced3.shape

# %%
df_metrics = pd.DataFrame()
for i in range(targetsreduced3.shape[1]):
    df_metrics['pca_'+str(i)] = targetsreduced3[:,i]
df_metrics.to_csv('../Data/df_metrics_newPCA_wave3.csv', index=False)

# %%
sns.set_context("talk")

sns.scatterplot(x=targetsreduced3[:,0],y=targetsreduced3[:,1],
                hue=targetsreduced3[:,0], palette='coolwarm', legend=False)
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*', label='Ground Truth')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# %% [markdown]
# ## wave4

# %%
inputs4 = pd.read_csv("../Data/exp_TuningL94_newPCA_wave4.csv").values

# %%
sns.set_context("talk")

plt.figure(figsize=(15,15))

plt.subplot(2,3,2)
plt.plot(inputs4[:,1], inputs4[:,2], ".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("c")
plt.legend()


plt.subplot(2,3,1)
plt.plot(inputs4[:,0], inputs4[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("F")
plt.legend()

plt.subplot(2,3,3)
plt.plot(inputs4[:,2], inputs4[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

plt.subplot(2,3,4)
plt.plot(inputs4[:,3], inputs4[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("b")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,5)
plt.plot(inputs4[:,2], inputs4[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("c")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,6)
plt.plot(inputs4[:,1], inputs4[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("b")
plt.legend()

# %%
# %%time

results4 = Parallel(n_jobs=20)(delayed(metric)(i) for i in inputs4)

targets4 = np.array(results4)

# %%
inputs4 = inputs4[np.all(np.isfinite(targets4), axis=1)]
targets4 = targets4[np.all(np.isfinite(targets4), axis=1)]

# %%
inputs4.shape, targets4.shape

# %%
targetsreduced4 = reduceL96pca_usingwave1(targets4, sc, pc)
targetsreduced4.shape

# %%
df_metrics = pd.DataFrame()
for i in range(targetsreduced4.shape[1]):
    df_metrics['pca_'+str(i)] = targetsreduced4[:,i]
df_metrics.to_csv('../Data/df_metrics_newPCA_wave4.csv', index=False)

# %%
sns.set_context("talk")

sns.scatterplot(x=targetsreduced4[:,0],y=targetsreduced4[:,1],
                hue=targetsreduced4[:,0], palette='coolwarm', legend=False)
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*', label='Ground Truth')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# %% [markdown]
# ## wave5

# %%
inputs5 = pd.read_csv("../Data/exp_TuningL94_newPCA_wave5.csv").values

# %%
sns.set_context("talk")

plt.figure(figsize=(15,15))

plt.subplot(2,3,2)
plt.plot(inputs5[:,1], inputs5[:,2], ".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("c")
plt.legend()


plt.subplot(2,3,1)
plt.plot(inputs5[:,0], inputs5[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("F")
plt.legend()

plt.subplot(2,3,3)
plt.plot(inputs5[:,2], inputs5[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

plt.subplot(2,3,4)
plt.plot(inputs5[:,3], inputs5[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("b")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,5)
plt.plot(inputs5[:,2], inputs5[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("c")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,6)
plt.plot(inputs5[:,1], inputs5[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("b")
plt.legend()

# %%
# %%time

results5 = Parallel(n_jobs=20)(delayed(metric)(i) for i in inputs5)

targets5 = np.array(results5)

# %%
inputs5 = inputs5[np.all(np.isfinite(targets5), axis=1)]
targets5 = targets5[np.all(np.isfinite(targets5), axis=1)]

# %%
inputs5.shape, targets5.shape

# %%
targetsreduced5 = reduceL96pca_usingwave1(targets5, sc, pc)
targetsreduced5.shape

# %%
df_metrics = pd.DataFrame()
for i in range(targetsreduced5.shape[1]):
    df_metrics['pca_'+str(i)] = targetsreduced5[:,i]
df_metrics.to_csv('../Data/df_metrics_newPCA_wave5.csv', index=False)

# %%
sns.set_context("talk")

sns.scatterplot(x=targetsreduced5[:,0],y=targetsreduced5[:,1],
                hue=targetsreduced5[:,0], palette='coolwarm', legend=False)
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*', label='Ground Truth')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# %% [markdown]
# ## wave6

# %%
inputs6 = pd.read_csv("../Data/exp_TuningL94_newPCA_wave6.csv").values

# %%
sns.set_context("talk")

plt.figure(figsize=(15,15))

plt.subplot(2,3,2)
plt.plot(inputs6[:,1], inputs6[:,2], ".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("c")
plt.legend()


plt.subplot(2,3,1)
plt.plot(inputs6[:,0], inputs6[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("F")
plt.legend()

plt.subplot(2,3,3)
plt.plot(inputs6[:,2], inputs6[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

plt.subplot(2,3,4)
plt.plot(inputs6[:,3], inputs6[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("b")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,5)
plt.plot(inputs6[:,2], inputs6[:,0],".", label="Simulation Points")
plt.plot(10, 1,"d", label="true")
plt.xlabel("c")
plt.ylabel("h")
plt.legend()

plt.subplot(2,3,6)
plt.plot(inputs6[:,1], inputs6[:,3],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("F")
plt.ylabel("b")
plt.legend()

# %%
# %%time

results6 = Parallel(n_jobs=20)(delayed(metric)(i) for i in inputs6)

targets6 = np.array(results6)

# %%
inputs6 = inputs6[np.all(np.isfinite(targets6), axis=1)]
targets6 = targets6[np.all(np.isfinite(targets6), axis=1)]

# %%
inputs6.shape, targets6.shape

# %%
targetsreduced6 = reduceL96pca_usingwave1(targets6, sc, pc)
targetsreduced6.shape

# %%
df_metrics = pd.DataFrame()
for i in range(targetsreduced6.shape[1]):
    df_metrics['pca_'+str(i)] = targetsreduced6[:,i]
df_metrics.to_csv('../Data/df_metrics_newPCA_wave6.csv', index=False)

# %%
sns.set_context("talk")

sns.scatterplot(x=targetsreduced6[:,0],y=targetsreduced6[:,1],
                hue=targetsreduced6[:,0], palette='coolwarm', legend=False)
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*', label='Ground Truth')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()


# %% [markdown]
# # Simulate the L96 configurations

# %%
def simulate(params):
    l96param_spinup = L96TwoLevelOriginal(K=36,
                           save_dt=0.001,
                           X_init=X_init,
                           h=params[0],
                           F=params[1],
                           c=params[2],
                           b=params[3],
                           integration_type='coupled') 
    l96param_spinup.iterate(10)
    l96param = L96TwoLevelOriginal(K=36,
                           save_dt=0.001,
                           X_init=l96param_spinup.history.X[-1,:].values,
                           h=params[0],
                           F=params[1],
                           c=params[2],
                           b=params[3],
                           integration_type='coupled') 
    l96param.iterate(100)
    return l96param


# %%
configs = pd.read_csv("../Data/finaltestpoints_newPCA.csv").values
configs.tolist()

# %%
simus = [simulate(i) for i in configs.tolist()]

# %%
simus_metrics = [ii.mean_stats(ax=0) for ii in simus]

# %%
simus_metrics = np.array(simus_metrics)
simus_metrics.shape

# %%
simusreduced = reduceL96pca_usingwave1(simus_metrics, sc, pc)
simusreduced.shape

# %%
simus_df = pd.DataFrame()

for i in range(simusreduced.shape[1]):
    simus_df['PC'+str(i+1)] = simusreduced[:,i]
    
#simus_df.to_csv('../Data/configs_metrics_newPCA.csv', index=False)

# %%
simus_df

# %% [markdown]
# ### add obs vector

# %%
simus_df.loc[len(simus_df)]=list(Obsreduced[0,:])

# %%
simus_df

# %%
simusdftranspose = simus_df.T
simusdftranspose.columns = ['config1', 'config2', 'config3', 'config4', 'config5', 'Obs']
simusdftranspose

# %%
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1.5})

simusdftranspose.plot()

# %%
sns.set_style("white")
sns.set_context("poster", font_scale=0.8, rc={"lines.linewidth": 3})

figure, axes = plt.subplots(2, 3, figsize=(25,10))
simusdftranspose['config1'].plot(ax=axes[0,0])
simusdftranspose['config2'].plot(ax=axes[0,0])
simusdftranspose['config3'].plot(ax=axes[0,0])
simusdftranspose['config4'].plot(ax=axes[0,0])
simusdftranspose['config5'].plot(ax=axes[0,0])
simusdftranspose['Obs'].plot(linestyle="--", ax=axes[0,0])
#axes[0,0].legend()

sns.kdeplot(simus[0].history.X.mean(axis=1), label='config1', ax=axes[0,1])
sns.kdeplot(simus[1].history.X.mean(axis=1), label='config2', ax=axes[0,1])
sns.kdeplot(simus[2].history.X.mean(axis=1), label='config3', ax=axes[0,1])
sns.kdeplot(simus[3].history.X.mean(axis=1), label='config4', ax=axes[0,1])
sns.kdeplot(simus[4].history.X.mean(axis=1), label='config5', ax=axes[0,1])
sns.kdeplot(l96_two_truth.history.X.mean(axis=1), label='Obs', color='tab:brown', linestyle="--", ax=axes[0,1])
#axes[0,1].legend()

sns.kdeplot(simus[0].history.Y_mean.mean(axis=1), label='config1', ax=axes[0,2])
sns.kdeplot(simus[1].history.Y_mean.mean(axis=1), label='config2', ax=axes[0,2])
sns.kdeplot(simus[2].history.Y_mean.mean(axis=1), label='config3', ax=axes[0,2])
sns.kdeplot(simus[3].history.Y_mean.mean(axis=1), label='config4', ax=axes[0,2])
sns.kdeplot(simus[4].history.Y_mean.mean(axis=1), label='config5', ax=axes[0,2])
sns.kdeplot(l96_two_truth.history.Y_mean.mean(axis=1), label='Obs', linestyle="--", ax=axes[0,2])
#axes[0,2].legend()
axes[0,2].set_xlabel(r'$\bar{Y}$')

sns.kdeplot((simus[0].history.X**2).mean(axis=1), label='config1', ax=axes[1,0])
sns.kdeplot((simus[1].history.X**2).mean(axis=1), label='config2', ax=axes[1,0])
sns.kdeplot((simus[2].history.X**2).mean(axis=1), label='config3', ax=axes[1,0])
sns.kdeplot((simus[3].history.X**2).mean(axis=1), label='config4', ax=axes[1,0])
sns.kdeplot((simus[4].history.X**2).mean(axis=1), label='config5', ax=axes[1,0])
sns.kdeplot((l96_two_truth.history.X**2).mean(axis=1), label='Obs', linestyle="--", ax=axes[1,0])
#axes[1,0].legend()
axes[1,0].set_xlabel(r'$X^2$')

sns.kdeplot((simus[0].history.X*simus[0].history.Y_mean).mean(axis=1), label='config1', ax=axes[1,1])
sns.kdeplot((simus[1].history.X*simus[1].history.Y_mean).mean(axis=1), label='config2', ax=axes[1,1])
sns.kdeplot((simus[2].history.X*simus[2].history.Y_mean).mean(axis=1), label='config3', ax=axes[1,1])
sns.kdeplot((simus[3].history.X*simus[3].history.Y_mean).mean(axis=1), label='config4', ax=axes[1,1])
sns.kdeplot((simus[4].history.X*simus[4].history.Y_mean).mean(axis=1), label='config5', ax=axes[1,1])
sns.kdeplot((l96_two_truth.history.X*l96_two_truth.history.Y_mean).mean(axis=1), label='Obs', linestyle="--", ax=axes[1,1])
#axes[1,1].legend()
axes[1,1].set_xlabel(r'$X\bar{Y}$')

sns.kdeplot(simus[0].history.Y2_mean.mean(axis=1), label='config1', ax=axes[1,2])
sns.kdeplot(simus[1].history.Y2_mean.mean(axis=1), label='config2', ax=axes[1,2])
sns.kdeplot(simus[2].history.Y2_mean.mean(axis=1), label='config3', ax=axes[1,2])
sns.kdeplot(simus[3].history.Y2_mean.mean(axis=1), label='config4', ax=axes[1,2])
sns.kdeplot(simus[4].history.Y2_mean.mean(axis=1), label='config5', ax=axes[1,2])
sns.kdeplot(l96_two_truth.history.Y2_mean.mean(axis=1), label='Obs', linestyle="--", ax=axes[1,2])
#axes[1,2].legend()
axes[1,2].set_xlabel(r'$\bar{Y^2}$')

handles, labels = axes[1,2].get_legend_handles_labels()
figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
          ncol=6, fancybox=True, shadow=False)

# %%
sns.set_style("white")
sns.set_context("poster", font_scale=0.8, rc={"lines.linewidth": 3})

figure, axes = plt.subplots(1, 5, figsize=(20,4))

sns.kdeplot(simus[0].history.X.mean(axis=1), label='config1', ax=axes[0])
sns.kdeplot(simus[1].history.X.mean(axis=1), label='config2', ax=axes[0])
sns.kdeplot(simus[2].history.X.mean(axis=1), label='config3', ax=axes[0])
sns.kdeplot(simus[3].history.X.mean(axis=1), label='config4', ax=axes[0])
sns.kdeplot(l96_two_truth.history.X.mean(axis=1), label='Obs', color='tab:brown', linestyle="--", ax=axes[0])
#axes[0].legend()#loc='upper center', bbox_to_anchor=(0.5, 1.05),
          #ncol=3, fancybox=True, shadow=True)
axes[0].set_xticks([])
axes[0].set_yticks([])

sns.kdeplot(simus[0].history.Y_mean.mean(axis=1), label='config1', ax=axes[1])
sns.kdeplot(simus[1].history.Y_mean.mean(axis=1), label='config2', ax=axes[1])
sns.kdeplot(simus[2].history.Y_mean.mean(axis=1), label='config3', ax=axes[1])
sns.kdeplot(simus[3].history.Y_mean.mean(axis=1), label='config4', ax=axes[1])
sns.kdeplot(l96_two_truth.history.Y_mean.mean(axis=1), label='Obs', color='tab:brown', linestyle="--", ax=axes[1])
#axes[1].legend()
axes[1].set_xlabel(r'$\bar{Y}$')
axes[1].set_xticks([])
axes[1].set_yticks([])

sns.kdeplot((simus[0].history.X**2).mean(axis=1), label='config1', ax=axes[2])
sns.kdeplot((simus[1].history.X**2).mean(axis=1), label='config2', ax=axes[2])
sns.kdeplot((simus[2].history.X**2).mean(axis=1), label='config3', ax=axes[2])
sns.kdeplot((simus[3].history.X**2).mean(axis=1), label='config4', ax=axes[2])
sns.kdeplot((l96_two_truth.history.X**2).mean(axis=1), label='Obs', color='tab:brown', linestyle="--", ax=axes[2])
#axes[2].legend()
axes[2].set_xlabel(r'$X^2$')
axes[2].set_xticks([])
axes[2].set_yticks([])

sns.kdeplot((simus[0].history.X*simus[0].history.Y_mean).mean(axis=1), label='config1', ax=axes[3])
sns.kdeplot((simus[1].history.X*simus[1].history.Y_mean).mean(axis=1), label='config2', ax=axes[3])
sns.kdeplot((simus[2].history.X*simus[2].history.Y_mean).mean(axis=1), label='config3', ax=axes[3])
sns.kdeplot((simus[3].history.X*simus[3].history.Y_mean).mean(axis=1), label='config4', ax=axes[3])
sns.kdeplot((l96_two_truth.history.X*l96_two_truth.history.Y_mean).mean(axis=1), label='Obs', color='tab:brown', linestyle="--", ax=axes[3])
#axes[3].legend()
axes[3].set_xlabel(r'$X\bar{Y}$')
axes[3].set_xticks([])
axes[3].set_yticks([])

sns.kdeplot(simus[0].history.Y2_mean.mean(axis=1), label='config1', ax=axes[4])
sns.kdeplot(simus[1].history.Y2_mean.mean(axis=1), label='config2', ax=axes[4])
sns.kdeplot(simus[2].history.Y2_mean.mean(axis=1), label='config3', ax=axes[4])
sns.kdeplot(simus[3].history.Y2_mean.mean(axis=1), label='config4', ax=axes[4])
sns.kdeplot(l96_two_truth.history.Y2_mean.mean(axis=1), label='Obs', color='tab:brown', linestyle="--", ax=axes[4])
#axes[4].legend()
axes[4].set_xlabel(r'$\bar{Y^2}$')
axes[4].set_xticks([])
axes[4].set_yticks([])

handles, labels = axes[4].get_legend_handles_labels()
figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=5, fancybox=True, shadow=False)

# %%
from scipy.stats import entropy

def KL_div_obs(sim):
    rangex=(l96_two_truth.history.X.mean(axis=1).min().values,
       l96_two_truth.history.X.mean(axis=1).max().values)
    rangeym=(l96_two_truth.history.Y_mean.mean(axis=1).min().values,
       l96_two_truth.history.Y_mean.mean(axis=1).max().values)
    rangexsq=((l96_two_truth.history.X**2).mean(axis=1).min().values,
       (l96_two_truth.history.X**2).mean(axis=1).max().values)
    rangexym=((l96_two_truth.history.X*l96_two_truth.history.Y_mean).mean(axis=1).min().values,
       (l96_two_truth.history.X*l96_two_truth.history.Y_mean).mean(axis=1).max().values)
    rangeymsq=(l96_two_truth.history.Y2_mean.mean(axis=1).min().values,
       l96_two_truth.history.Y2_mean.mean(axis=1).max().values)
    return np.median([entropy(np.histogram(sim.history.X.mean(axis=1), range=rangex)[0],
                          np.histogram(l96_two_truth.history.X.mean(axis=1))[0]),
                  entropy(np.histogram(sim.history.Y_mean.mean(axis=1), range=rangeym)[0],
                          np.histogram(l96_two_truth.history.Y_mean.mean(axis=1))[0]),
                  entropy(np.histogram((sim.history.X**2).mean(axis=1), range=rangexsq)[0],
                          np.histogram((l96_two_truth.history.X**2).mean(axis=1))[0]),
                  entropy(np.histogram((sim.history.X*sim.history.Y_mean).mean(axis=1), range=rangexym)[0],
                          np.histogram((l96_two_truth.history.X*l96_two_truth.history.Y_mean).mean(axis=1))[0]),
                  entropy(np.histogram(sim.history.Y2_mean.mean(axis=1), range=rangeymsq)[0],
                          np.histogram(l96_two_truth.history.Y2_mean.mean(axis=1))[0])])


# %%
KL_div_obs(l96_two_truth)

# %%
[KL_div_obs(sim) for sim in simus]

# %%
[KL_div_obs(sim) for sim in simus]

# %%
configs
