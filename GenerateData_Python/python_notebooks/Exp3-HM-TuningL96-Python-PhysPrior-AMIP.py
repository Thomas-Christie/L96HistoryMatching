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
# \bar{Y} \\
# \bar{Y}^{2}
# \end{array}\right)
# $$

# %% [markdown]
# # History Matching

# %% [markdown]
# We want to find the parameters (h, c, b)
#
# $ \frac{d Y_{j, k}}{d t}=\underbrace{-b c Y_{j+1, k}\left(Y_{j+2, k}-Y_{j-1, k}\right)}_{\text {Advection }} \underbrace{- c Y_{j, k}}_{\text {Diffusion }} \underbrace{+\frac{h c}{b} X_{k}}_{\text {Coupling }}$

# %% [markdown]
# # Wave1

# %% [markdown]
# calculate metrics for the true trajectory

# %%
TrueTraj = l96_two_truth.mean_stats(ax=0)[list(range(36,36*2))+list(range(36*4,36*5)),]


# %% [markdown]
# Define metric 

# %%
def metric(params, Xdata):
    l96param = AMIP(Xdata=Xdata,
                    K=36,
                    save_dt=0.001,
                    X_init=Xdata[0,:],
                    h=params[0],
                    F=10,
                    c=params[1],
                    b=params[2]) 
    l96param.iterate(100)
    return l96param.mean_stats(ax=0)[list(range(36,36*2))+list(range(36*4,36*5)),]


# %% [markdown]
# Sanity check with the true params

# %%
sancheck = metric(np.array([1,10,10]), l96_two_truth.history.X)

# %%
np.mean((TrueTraj - sancheck)**2)

# %% [markdown]
# # Running the 30 simulations in parallel

# %%
inputs = pd.read_csv("../Data/df_inputs_newPCA_PhysPrior_AMIP_wave1.csv").values

# %%
inputs

# %%
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.plot(inputs[:,0], inputs[:,2],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("b")
plt.legend()

plt.subplot(1,3,2)
plt.plot(inputs[:,0], inputs[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("c")
plt.legend()

plt.subplot(1,3,3)
plt.plot(inputs[:,1], inputs[:,2],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
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

results = Parallel(n_jobs=20)(delayed(metric)(i, l96_two_truth.history.X) for i in inputs)

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
plt.scatter(targetsreduced[:,0],targetsreduced[:,1], color='k')
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*')

# %% [markdown]
# ### save dataframes

# %%
df_metrics = pd.DataFrame()
for i in range(targetsreduced.shape[1]):
    df_metrics['pca_'+str(i)] = targetsreduced[:,i]
df_metrics.to_csv('../Data/df_metrics_newPCA_PhysPrior_AMIP_classic_wave1.csv', index=False)

# %%
df_metrics

# %%
df_obs_nonoise = pd.DataFrame()
for i in range(targetsreduced.shape[1]):
    df_obs_nonoise['pca_'+str(i)] = Obsreduced[:,i]
df_obs_nonoise.to_csv('../Data/df_obs_nonoise_newPCA_PhysPrior_AMIP_classic_wave1.csv', index=False)

# %% [markdown]
# ## 2) wave 2

# %%
inputs2 = pd.read_csv("../Data/exp_TuningL94_newPCA_PhysPrior_AMIP_classic_wave2.csv").values

# %%
sns.set_context("talk")

plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.plot(inputs2[:,0], inputs2[:,2],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("b")
plt.legend()

plt.subplot(1,3,2)
plt.plot(inputs2[:,0], inputs2[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("c")
plt.legend()

plt.subplot(1,3,3)
plt.plot(inputs2[:,1], inputs2[:,2],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

# %%
inputs2.shape

# %%
# %%time

results2 = Parallel(n_jobs=20)(delayed(metric)(i, l96_two_truth.history.X) for i in inputs2)

targets2 = np.array(results2)

# %%
inputs2 = inputs2[np.all(np.isfinite(targets2), axis=1)]
targets2 = targets2[np.all(np.isfinite(targets2), axis=1)]


# %%
def reduceL96pca_usingwave1(vector, scaler, pca): 
    datascaled = scaler.transform(vector)
    return pca.transform(datascaled)


# %%
#targetsreduced2, Obsreduced2, sc2, pc2 = reduceL96pca(targets2)
targetsreduced2 = reduceL96pca_usingwave1(targets2, sc, pc)
targetsreduced2.shape

# %%
plt.scatter(targetsreduced2[:,0],targetsreduced2[:,1], color='k')
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*')

# %%
df_metrics = pd.DataFrame()
for i in range(targetsreduced2.shape[1]):
    df_metrics['pca_'+str(i)] = targetsreduced2[:,i]
df_metrics.to_csv('../Data/df_metrics_newPCA_PhysPrior_AMIP_classic_wave2.csv', index=False)

# %% [markdown]
# # wave3

# %%
inputs3 = pd.read_csv("../Data/exp_TuningL94_newPCA_PhysPrior_AMIP_classic_wave3.csv").values

# %%
sns.set_context("talk")

plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.plot(inputs3[:,0], inputs3[:,2],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("b")
plt.legend()

plt.subplot(1,3,2)
plt.plot(inputs3[:,0], inputs3[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("c")
plt.legend()

plt.subplot(1,3,3)
plt.plot(inputs3[:,1], inputs3[:,2],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

# %%
inputs3.shape

# %%
# %%time

results3 = Parallel(n_jobs=20)(delayed(metric)(i, l96_two_truth.history.X) for i in inputs3)

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
df_metrics.to_csv('../Data/df_metrics_newPCA_PhysPrior_AMIP_classic_wave3.csv', index=False)

# %%
plt.scatter(targetsreduced3[:,0],targetsreduced3[:,1], color='k')
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*')

# %% [markdown]
# # wave4

# %%
inputs4 = pd.read_csv("../Data/exp_TuningL94_newPCA_PhysPrior_AMIP_classic_wave4.csv").values

# %%
sns.set_context("talk")

plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.plot(inputs4[:,0], inputs4[:,2],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("b")
plt.legend()

plt.subplot(1,3,2)
plt.plot(inputs4[:,0], inputs4[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("c")
plt.legend()

plt.subplot(1,3,3)
plt.plot(inputs4[:,1], inputs4[:,2],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

# %%
inputs4.shape

# %%
# %%time

results4 = Parallel(n_jobs=20)(delayed(metric)(i, l96_two_truth.history.X) for i in inputs4)

targets4 = np.array(results4)

# %%
inputs4.shape, targets4.shape

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
df_metrics.to_csv('../Data/df_metrics_newPCA_PhysPrior_AMIP_classic_wave4.csv', index=False)

# %%
plt.scatter(targetsreduced4[:,0],targetsreduced4[:,1], color='k')
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*')

# %% [markdown]
# # wave5

# %%
inputs5 = pd.read_csv("../Data/exp_TuningL94_newPCA_PhysPrior_AMIP_classic_wave5.csv").values

# %%
sns.set_context("talk")

plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.plot(inputs5[:,0], inputs5[:,2],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("b")
plt.legend()

plt.subplot(1,3,2)
plt.plot(inputs5[:,0], inputs5[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("h")
plt.ylabel("c")
plt.legend()

plt.subplot(1,3,3)
plt.plot(inputs5[:,1], inputs5[:,2],".", label="Simulation Points")
plt.plot(10, 10,"d", label="true")
plt.xlabel("c")
plt.ylabel("b")
plt.legend()

# %%
inputs5.shape

# %%
# %%time

results5 = Parallel(n_jobs=20)(delayed(metric)(i, l96_two_truth.history.X) for i in inputs5)

targets5 = np.array(results5)

# %%
inputs5.shape, targets5.shape

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
df_metrics.to_csv('../Data/df_metrics_newPCA_PhysPrior_AMIP_classic_wave5.csv', index=False)

# %%
plt.scatter(targetsreduced5[:,0],targetsreduced5[:,1], color='k')
plt.plot(Obsreduced[:,0],Obsreduced[:,1], 'r*')


# %% [markdown]
# # Simulate the L96 configurations

# %%
def simulate(params, Xdata):
    l96param = AMIP(Xdata=Xdata,
                    K=36,
                    save_dt=0.001,
                    X_init=Xdata[0,:],
                    h=params[0],
                    F=10,
                    c=params[1],
                    b=params[2]) 
    l96param.iterate(100)
    return l96param


# %%
configs = pd.read_csv("/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/finaltestpoints_newPCA_PhysPrior_classic_AMIP.csv").values
configs.tolist()

# %%
#pd.read_csv("/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/finaltestpoints_newPCA.csv").to_latex()

# %%
simus = [simulate(i, l96_two_truth.history.X) for i in configs.tolist()]

# %%
simus_metrics = [ii.mean_stats(ax=0)[list(range(36,36*2))+list(range(36*4,36*5)),] for ii in simus]

# %%
simus_metrics = np.array(simus_metrics)
simus_metrics.shape

# %%
simusreduced = reduceL96pca_usingwave1(simus_metrics, sc, pc)
simusreduced.shape

# %%
simus_df = pd.DataFrame()

for i in range(simusreduced.shape[1]):
    simus_df['pc'+str(i)] = simusreduced[:,i]
    
#simus_df.to_csv('/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/configs_metrics_newPCA_PhysPrior_classic_AMIP.csv', index=False)

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
simusdftranspose.columns = ['config1', 'config2','Obs']
simusdftranspose

# %%
sns.set_style("white")
sns.set_context("poster", font_scale=0.5, rc={"lines.linewidth": 3})

figure, axes = plt.subplots(1, 3, figsize=(16,6))
simusdftranspose['config1'].plot(ax=axes[0])
simusdftranspose['config2'].plot(ax=axes[0])
simusdftranspose['Obs'].plot(linestyle="--", color='tab:brown', ax=axes[0])
axes[0].set_xticks(simusdftranspose.index)
axes[0].legend()

sns.kdeplot(simus[0].history.Y_mean.mean(axis=1), label='config1', ax=axes[1])
sns.kdeplot(simus[1].history.Y_mean.mean(axis=1), label='config2', ax=axes[1])
sns.kdeplot(l96_two_truth.history.Y_mean.mean(axis=1), label='Obs', color='tab:brown', linestyle="--", ax=axes[1])
axes[1].legend()
axes[1].set_xlabel(r'$\bar{Y}$')

sns.kdeplot(simus[0].history.Y2_mean.mean(axis=1), label='config1', ax=axes[2])
sns.kdeplot(simus[1].history.Y2_mean.mean(axis=1), label='config2', ax=axes[2])
sns.kdeplot(l96_two_truth.history.Y2_mean.mean(axis=1), label='Obs', color='tab:brown', linestyle="--", ax=axes[2])
axes[2].legend()
axes[2].set_xlabel(r'$\bar{Y^2}$')

# %%
bx_df_Ymean = pd.DataFrame()
bx_df_Ymean['config0'] = simus[0].history.Y_mean.mean(axis=0)
bx_df_Ymean['config1'] = simus[1].history.Y_mean.mean(axis=0)
bx_df_Ymean['config2'] = simus[2].history.Y_mean.mean(axis=0)
bx_df_Ymean['config3'] = simus[3].history.Y_mean.mean(axis=0)

bx_df_Ymean['AvgConfig'] = 0.25*(simus[0].history.Y_mean.mean(axis=0)+
                                 simus[1].history.Y_mean.mean(axis=0)+
                                simus[2].history.Y_mean.mean(axis=0)+
                                simus[3].history.Y_mean.mean(axis=0))
bx_df_Ymean['Obs'] = TrueTraj[0:36*1,]
bx_df_Ymean.boxplot()

# %%
from scipy.stats import entropy

def KL_div_obs(sim):
    rangeym=(l96_two_truth.history.Y_mean.mean(axis=1).min().values,
       l96_two_truth.history.Y_mean.mean(axis=1).max().values)
    rangeymsq=(l96_two_truth.history.Y2_mean.mean(axis=1).min().values,
       l96_two_truth.history.Y2_mean.mean(axis=1).max().values)
    return np.median([entropy(np.histogram(sim.history.Y_mean.mean(axis=1), range=rangeym)[0],
                          np.histogram(l96_two_truth.history.Y_mean.mean(axis=1))[0]),
                  entropy(np.histogram(sim.history.Y2_mean.mean(axis=1), range=rangeymsq)[0],
                          np.histogram(l96_two_truth.history.Y2_mean.mean(axis=1))[0])])

[KL_div_obs(sim) for sim in simus]

# %%
