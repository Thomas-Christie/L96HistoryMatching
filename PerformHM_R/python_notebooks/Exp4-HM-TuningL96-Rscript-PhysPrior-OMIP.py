# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: R [conda env:.conda-r_redouane]
#     language: R
#     name: conda-env-.conda-r_redouane-r
# ---

# %% [markdown]
# ## Libraries

# %%
mogp_dir <- "/gpfs7kw/linkhome/rech/genloc01/udu91zn/mogp_emulator"
## important to add "import warning" at MutliOutputGP.py

source('/gpfswork/rech/omr/udu91zn/HighTune_R/BuildEmulator/BuildEmulator.R')
source('/gpfswork/rech/omr/udu91zn/HighTune_R/HistoryMatching/HistoryMatching.R')
#source("HistoryMatching/impLayoutplot.R")
source('/gpfswork/rech/omr/udu91zn/HighTune_R/BuildEmulator/utils.R')

library(comprehenr)
library(caret)
library("future.apply") #important for parallel version of ImplausibilityMOGP function in HistoryMatching.R

# %% [markdown]
# # Read data 

# %%
my_bounds <- data.frame(c(0, 20), c(0, 20))

# %%
set.seed(42)
inputs <- as.data.frame(2*maximinLHS(20, 2)-1)
inputs_unscaled <- rangeUnscale(inputs, my_bounds)
names(inputs) <- c('G','F')
names(inputs_unscaled) <- c('G','F')
inputs_unscaled

# %%
write.csv(inputs_unscaled,"/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/df_inputs_newPCA_PhysPrior_OMIP_wave1.csv", row.names = FALSE)

# %%
set.seed(42)
                       
#Load outputs and select variables you want to keep
outputs <- read.csv("/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/df_metrics_newPCA_PhysPrior_OMIP_wave1.csv")
                         
#Add some noise (here no noise added)
N = nrow(inputs) #nb samples 
noise <- rnorm(N, 0, 0.5)
tData <- cbind(inputs, noise, outputs)
names(tData)[names(tData) == "noise"] <- "Noise"
                         
head(tData)

# %% [markdown]
# ### Training Emulators

# %%
#choices.new <- choices.default
#choices.new$lm.maxdf = 3 ### NOT WORKING
#choices.new$lm.tryFouriers=TRUE  ### NOT WORKING

# %%
TestEm <- BuildNewEmulators(tData,
                            HowManyEmulators = length(outputs),
                            meanFun = "fitted",
                            #kernel = c("Matern52"),
                            additionalVariables = names(tData)[1:2])  #important to put this line
                            #Choices = lapply(1:length(outputs), function(k) choices.new))

# %%
tObs <- read.csv("/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/df_obs_nonoise_newPCA_PhysPrior_OMIP_wave1.csv")
tObs <- as.vector(t(tObs))

tDisc <- rep(0, length(outputs)) ### variances
#apply(tObs[1:2] - TestEm$mogp$predict(array(c(inputs$G, inputs$F), dim=c(20,2)))$mean, 1 , var)
tObsErr <- rep(0, length(outputs))

# %% [markdown]
# ## LOO

# %%
cands <- names(tData)[1:2]

# %%
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 1, ParamNames = cands, Obs = tObs[1], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 2, ParamNames = cands, Obs = tObs[2], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 3, ParamNames = cands, Obs = tObs[3], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 4, ParamNames = cands, Obs = tObs[4], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 5, ParamNames = cands, Obs = tObs[5], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 6, ParamNames = cands, Obs = tObs[6], ObsErr = 0.)#tLOOs <- 

# %% [markdown]
# ## History Matching
# * Observations discrepancy variances in *tDisc* (here sets to 0)
# * Observations errors in *tObsErr* (here set to 0)

# %%
future::availableCores() 

# %%
set.seed(42)
sample_size <- 1000000
nparam <- length(names(TestEm$fitting.elements$Design))
Xp <- as.data.frame(2*randomLHS(sample_size, nparam)-1)
names(Xp) <- names(TestEm$fitting.elements$Design)

# %%
system.time(Timps <- ImplausibilityMOGP(NewData=Xp, Emulator=TestEm, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr))

# %%
ImpData_wave1 = cbind(Xp, Timps)
print(object.size(ImpData_wave1), units="Mb")

# %%
VarNames <- names(Xp)
valmax = 0 #how many outputs can be above the implausibility cut off?
cutoff_vec <- 3 #the implausibility cut off

param.def = data.frame(G=c(1), F=c(10)) #the default parameters of the model (on [-1,1])
param.defaults.norm = rangeScale(param.def, my_bounds, range(-1,1))
print(param.defaults.norm)

# %%
ImpListM1 = CreateImpList(whichVars = 1:nparam, VarNames=VarNames, ImpData=ImpData_wave1, nEms=TestEm$mogp$n_emulators, whichMax=valmax+1)
NROY1 <- which(rowSums(Timps <= cutoff_vec[1]) >= TestEm$mogp$n_emulators -valmax)
ratio1 <- length(NROY1)/dim(Xp)[1]
ratio1

# %% [markdown]
# #### 20 samples for training, 1 000 000 points for test

# %%
imp.layoutm11(ImpListM1,VarNames,VariableDensity=FALSE,newPDF=FALSE,the.title=paste("InputSpace_wave",WAVEN,".pdf",sep=""),newPNG=FALSE,newJPEG=FALSE,newEPS=FALSE,Points=matrix(param.defaults.norm,ncol=nparam))
mtext(paste("Remaining space:",length(NROY1)/dim(Xp)[1],sep=""), side=1)

# %%
indminImp <- order(apply(Timps[NROY1,], 1, FUN=max))
rangeUnscale(Xp[NROY1[indminImp[1:10]],], my_bounds)

# %%
colMeans(rangeUnscale(Xp[NROY1,], my_bounds))

# %%
length(NROY1)

# %% [markdown]
# # Wave2

# %%
20/ratio1

# %%
set.seed(42)

designpoints <- data.frame()

while (nrow(designpoints) < 20) {
        tmp <- as.data.frame(2*maximinLHS(ceil(20/ratio1), 2)-1)
        names(tmp) <- names(TestEm$fitting.elements$Design)
        imps_tmp <- ImplausibilityMOGP(NewData=tmp, Emulator=TestEm, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp <- which(rowSums(imps_tmp <= cutoff_vec[1]) >= TestEm$mogp$n_emulators -valmax)
        selectionP <- tmp[NROYtmp,]
        row.names(selectionP) <- NULL ## to avoid double index
        designpoints <- rbind(designpoints,selectionP)
        print(nrow(designpoints))
        flush.console()
        } 

designpoints <- designpoints[sample(nrow(designpoints),20),]
row.names(designpoints) <- NULL

designpoints_denorm <- rangeUnscale(designpoints, my_bounds)

# %%
designpoints_denorm

# %%
write.csv(designpoints_denorm,"/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/exp_TuningL94_newPCA_PhysPrior_OMIP_wave2.csv", row.names = FALSE)

# %%
inputs <- designpoints

#Load outputs and select variables you want to keep
outputs <- read.csv("/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/df_metrics_newPCA_PhysPrior_OMIP_wave2.csv")
                         
#Add some noise (here no noise added)
set.seed(42)

N = nrow(inputs) #nb samples 
noise <- rnorm(N, 0, 0.5)
tData <- cbind(inputs, noise, outputs)
names(tData)[names(tData) == "noise"] <- "Noise"
                         
tData

# %%
TestEm2 <- BuildNewEmulators(tData,
                            HowManyEmulators = length(outputs),
                            meanFun = "fitted",
                            #kernel = c("Matern52"),
                            additionalVariables = names(tData)[1:2])  #important to put this line

                            #Choices = lapply(1:length(outputs),
                            #                   function(k) choices.new),

# %%
cands <- names(tData)[1:2]
tLOOs <- LOO.plot(Emulators = TestEm2, which.emulator = 1, ParamNames = cands, Obs = tObs[1], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm2, which.emulator = 2, ParamNames = cands, Obs = tObs[2], ObsErr = 0.)#tLOOs <- 

# %%
Timps2 <- matrix(rep(t(Timps),1), ncol=ncol(Timps), byrow=TRUE)
system.time(Timps2[NROY1,] <- ImplausibilityMOGP(NewData=Xp[NROY1,], Emulator=TestEm2, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr))

# %%
ImpData_wave2 = cbind(Xp, Timps2)

# %%
valmax2 = 0 #how many outputs can be above the implausibility cut off?
ImpListM2 = CreateImpList(whichVars = 1:nparam, VarNames=VarNames, ImpData=ImpData_wave2, nEms=TestEm2$mogp$n_emulators, whichMax=valmax2+1)
NROY2 <- which(rowSums(Timps2 <= cutoff_vec[1]) >= TestEm2$mogp$n_emulators -valmax2)
ratio2 <- length(NROY2)/dim(Xp)[1]
ratio2

# %%
imp.layoutm11(ImpListM2,VarNames,VariableDensity=FALSE,newPDF=FALSE,the.title=paste("InputSpace_wave",WAVEN,".pdf",sep=""),newPNG=FALSE,newJPEG=FALSE,newEPS=FALSE,Points=matrix(param.defaults.norm,ncol=nparam))
mtext(paste("Remaining space:",length(NROY2)/dim(Xp)[1],sep=""), side=1)

# %%
indminImp2 <- order(apply(Timps2[NROY2,], 1, FUN=max))
rangeUnscale(Xp[NROY2[indminImp2[1:10]],], my_bounds)

# %%
length(NROY2)

# %% [markdown]
# ## K-means

# %%
library(ClusterR)

# %%
preProcValues <- preProcess(Xp[NROY1,], method = c("center", "scale"))
normalizeddata <- predict(preProcValues, Xp[NROY1,])

# %%
opt_km = Optimal_Clusters_KMeans(normalizeddata, 
                                 criterion = "silhouette", 
                                 max_clusters=10,
                                 plot_clusters = TRUE)

# %%
classif <- kmeans(normalizeddata, centers=1, iter.max=100, nstart=100)
kmcenters <- unPreProc(preProcValues, data.frame(classif$centers))
candidates <- rangeUnscale(kmcenters, my_bounds)
candidates

# %%
nrow(rangeUnscale(unPreProc(preProcValues, normalizeddata[classif$cluster==2,]), my_bounds))

# %%
library(xtable)

print.xtable(xtable(candidates))

# %% [markdown]
# # Ensemble of plausible simulations

# %%
### check if Kmeans centers are in NROY4
imps_kmeans <- ImplausibilityMOGP(NewData=kmcenters, Emulator=TestEm5, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
which(rowSums(imps_kmeans <= cutoff_vec[1]) >= TestEm5$mogp$n_emulators -valmax4)

# %%
testpoints <- rangeUnscale(kmcenters, my_bounds)[rowSums(imps_kmeans <= cutoff_vec[1]) >= TestEm5$mogp$n_emulators -valmax5,]

# %%
testpoints

# %%
write.csv(testpoints,"/gpfswork/rech/omr/udu91zn/HighTune_R/Data/JAMES/finaltestpoints_newPCA_PhysPrior_classic_AMIP.csv", row.names = FALSE)

# %% [markdown]
# ## summary of the HM

# %%
NROYs <- 100 * c(ratio1, ratio2, ratio3, ratio4, ratio5)
NbSim <- c(30, nrow(designpoints), nrow(designpoints2), nrow(designpoints3), nrow(designpoints4))

# %%
data.frame(NROYs, NbSim)

# %%
print.xtable(xtable(data.frame(NROYs, NbSim)))

# %%
sum(NbSim)
