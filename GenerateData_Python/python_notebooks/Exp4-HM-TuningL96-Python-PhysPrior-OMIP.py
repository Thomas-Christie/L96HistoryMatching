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

# %%
l96_two_truth.history.X.plot()

# %% [markdown]
# This is our ground truth

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
TrueTraj = l96_two_truth.mean_stats(ax=0)[list(range(0,36))+list(range(36*2,36*4)),]


# %% [markdown]
# Define metric 

# %%
def metric(params, Ysumdata):
    l96param = OMIP(Ysumdata=Ysumdata,
                    K=36,
                    save_dt=0.001,
                    X_init = Newinit,#l96param_spinup.history.X[-1,:].values,
                    G=params[0],
                    F=params[1]) 
    l96param.iterate(100)
    return l96param.mean_stats(ax=0)[list(range(0,36))+list(range(36*2,36*4)),]


# %% [markdown]
# Sanity check with the true params

# %%
sancheck = metric(np.array([1.,10.]), l96_two_truth.history.Y_sum) 

# %%
#l96_two_truth.history.Y_sum-10*l96_two_truth.history.Y_mean

# %%
#sancheck.history.X.plot()
#plt.figure()
#l96_two_truth.history.X.plot()
#plt.figure()
#(sancheck.history.X-l96_two_truth.history.X).plot()

# %%
np.mean((TrueTraj - sancheck)**2) ### 0 with uncoupled

# %%
np.mean((TrueTraj - sancheck)**2) ### h=1/8

# %%
np.mean((TrueTraj - sancheck)**2) ### h=1/16

# %% [markdown]
# # Running the 20 simulations in parallel

# %%
inputs = pd.read_csv("/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/df_inputs_newPCA_PhysPrior_OMIP_wave1.csv").values

# %%
inputs

# %%
plt.figure(figsize=(10,5))
plt.plot(inputs[:,0], inputs[:,1],".", label="Simulation Points")
plt.plot(1, 10,"d", label="true")
plt.xlabel("G")
plt.ylabel("F")
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

results = Parallel(n_jobs=40)(delayed(metric)(i,l96_two_truth.history.Y_sum) for i in inputs)

targets = np.array(results)

# %%
inputs = inputs[np.all(np.isfinite(targets), axis=1)]
targets = targets[np.all(np.isfinite(targets), axis=1)]

# %%
targets.shape, inputs.shape

# %%
sns.set_style("white")
sns.set_context("poster", font_scale=0.5, rc={"lines.linewidth": 3})

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
for i in range(20):
    plt.plot(targets[i,:36], 'b')
plt.plot(targets[i,:36], 'b', label='LHS simulations') 
plt.plot(TrueTraj[:36], 'r--', label='Obs') 
plt.title('X metrics')
plt.xlabel(r'$X$ variable index')
plt.legend()

plt.subplot(1,3,2)
for i in range(20):
    plt.plot(targets[i,36:36*2], 'b')
plt.plot(targets[i,:36:36*2], 'b', label='LHS simulations') 
plt.plot(TrueTraj[36:36*2], 'r--', label='Obs')  
plt.title(r'$X^2$ metrics')
plt.xlabel(r'$X$ variable index')
plt.legend()

plt.subplot(1,3,3)
for i in range(20):
    plt.plot(targets[i,36*2:36*3], 'b')
plt.plot(targets[i,36*2:36*3], 'b', label='LHS simulations') 
plt.plot(TrueTraj[36*2:36*3], 'r--', label='Obs') 
plt.title(r'$X\bar{Y}$ metrics')
plt.xlabel(r'$X$ variable index')
plt.legend()

# %% [markdown]
# # pca

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def reduceL96pca(vector):   
    scaler = StandardScaler().fit(vector)
    datascaled = scaler.transform(vector)
    TrueTrajscaled = scaler.transform(TrueTraj[None,:2*36])#2*36
    ###########
    pca = PCA(n_components=0.99, svd_solver = 'full')
    pca.fit(datascaled)
    reduceddata = pca.transform(datascaled)
    reducedTrueTraj = pca.transform(TrueTrajscaled)
    return reduceddata, reducedTrueTraj, scaler, pca


# %%
targetsreduced, Obsreduced, sc, pc = reduceL96pca(targets[:,:2*36])#[:,:2*36]
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
df_metrics.to_csv('/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/df_metrics_newPCA_PhysPrior_OMIP_wave1.csv', index=False)

# %%
df_metrics

# %%
df_obs_nonoise = pd.DataFrame()
for i in range(targetsreduced.shape[1]):
    df_obs_nonoise['pca_'+str(i)] = Obsreduced[:,i]
df_obs_nonoise.to_csv('/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/df_obs_nonoise_newPCA_PhysPrior_OMIP_wave1.csv', index=False)
