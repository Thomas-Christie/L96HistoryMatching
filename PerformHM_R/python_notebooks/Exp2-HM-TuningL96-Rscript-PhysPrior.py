# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: R [conda env:envHM] *
#     language: R
#     name: conda-env-envHM-r
# ---

# %% [markdown]
# ## Libraries

# %%
mogp_dir <- "/data/rlguensat/conda_envs/envHM/lib/python3.8/site-packages/mogp_emulator"
## important to add "import warning" at MutliOutputGP.py

setwd('..')

source('ExeterUQ_MOGP/BuildEmulator/BuildEmulator.R')
source('ExeterUQ_MOGP/HistoryMatching/HistoryMatching.R')
#source("HistoryMatching/impLayoutplot.R")
source('ExeterUQ_MOGP/BuildEmulator/utils.R')

library(comprehenr)
library(caret, include.only = 'preProcess')
library("future.apply") #important for parallel version of ImplausibilityMOGP function in HistoryMatching.R

# %% [markdown]
# # Read data 

# %%
my_bounds <- data.frame(c(0, 2), c(0, 20), c(1, 20), c(1, 20))

# %%
set.seed(42)
inputs <- as.data.frame(2*maximinLHS(40, 4)-1)
inputs_unscaled <- rangeUnscale(inputs, my_bounds)
names(inputs) <- c('h','F','c','b')
names(inputs_unscaled) <- c('h','F','c','b')
inputs_unscaled

# %%
write.csv(inputs_unscaled,"Data/df_inputs_newPCA_PhysPrior_wave1.csv", row.names = FALSE)

# %%
set.seed(42)
                       
#Load outputs and select variables you want to keep
outputs <- read.csv("Data/df_metrics_newPCA_PhysPrior_wave1.csv")
                         
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
                            additionalVariables = names(tData)[1:4])  #important to put this line
                            #Choices = lapply(1:length(outputs), function(k) choices.new))

# %%
tDisc <- rep(0, length(outputs))
tObs <- read.csv("Data/df_obs_nonoise_newPCA_PhysPrior_wave1.csv")
tObs <- as.vector(t(tObs))
tObsErr <- rep(0, length(outputs))

# %% [markdown]
# ## LOO

# %%
cands <- names(tData)[1:4]

# %%
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 1, ParamNames = cands, Obs = tObs[1], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 2, ParamNames = cands, Obs = tObs[2], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 3, ParamNames = cands, Obs = tObs[3], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 4, ParamNames = cands, Obs = tObs[4], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm, which.emulator = 5, ParamNames = cands, Obs = tObs[5], ObsErr = 0.)#tLOOs <- 

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

param.def = data.frame(h=c(1), F=c(10), c=c(10),b=c(10)) #the default parameters of the model (on [-1,1])
param.defaults.norm = rangeScale(param.def, my_bounds, range(-1,1))
print(param.defaults.norm)

# %%
ImpListM1 = CreateImpList(whichVars = 1:nparam, VarNames=VarNames, ImpData=ImpData_wave1, nEms=TestEm$mogp$n_emulators, whichMax=valmax+1)
NROY1 <- which(rowSums(Timps <= cutoff_vec[1]) >= TestEm$mogp$n_emulators -valmax)
ratio1 <- length(NROY1)/dim(Xp)[1]
ratio1

# %% [markdown]
# #### 40 samples for training, 1 000 000 points for test

# %%
imp.layoutm11(ImpListM1,VarNames,VariableDensity=FALSE,newPDF=FALSE,the.title=paste("InputSpace_wave",WAVEN,".pdf",sep=""),newPNG=FALSE,newJPEG=FALSE,newEPS=FALSE,Points=matrix(param.defaults.norm,ncol=nparam))
mtext(paste("Remaining space:",length(NROY1)/dim(Xp)[1],sep=""), side=1)

# %%
indminImp <- order(apply(Timps[NROY1,], 1, FUN=max))
rangeUnscale(Xp[NROY1[indminImp[1:100]],], my_bounds)

# %%
length(NROY1)

# %% [markdown]
# # Wave2

# %%
40/ratio1

# %%
set.seed(42)

designpoints <- data.frame()

while (nrow(designpoints) <= 40) {
        tmp <- as.data.frame(2*maximinLHS(ceil(40/ratio1), 4)-1)
        names(tmp) <- names(TestEm$fitting.elements$Design)
        imps_tmp <- ImplausibilityMOGP(NewData=tmp, Emulator=TestEm, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp <- which(rowSums(imps_tmp <= cutoff_vec[1]) >= TestEm$mogp$n_emulators -valmax)
        selectionP <- tmp[NROYtmp,]
        row.names(selectionP) <- NULL ## to avoid double index
        designpoints <- rbind(designpoints,selectionP)
        print(nrow(designpoints))
        flush.console()
        } 

designpoints <- designpoints[sample(nrow(designpoints),40),]

designpoints_denorm <- rangeUnscale(designpoints, my_bounds)

# %%
designpoints_denorm

# %%
write.csv(designpoints_denorm,"Data/exp_TuningL94_newPCA_PhysPrior_wave2.csv", row.names = FALSE)

# %%
inputs <- designpoints

#Load outputs and select variables you want to keep
outputs <- read.csv("Data/df_metrics_newPCA_PhysPrior_wave2.csv")
                         
#Add some noise (here no noise added)
set.seed(42)

N = nrow(inputs) #nb samples 
noise <- rnorm(N, 0, 0.5)
tData <- cbind(inputs, noise, outputs)
names(tData)[names(tData) == "noise"] <- "Noise"
                         
head(tData)

# %%
TestEm2 <- BuildNewEmulators(tData,
                            HowManyEmulators = length(outputs),
                            meanFun = "fitted",
                            #kernel = c("Matern52"),
                            additionalVariables = names(tData)[1:4])  #important to put this line

                            #Choices = lapply(1:length(outputs),
                            #                   function(k) choices.new),

# %%
tObs2 <- read.csv("Data/df_obs_nonoise_newPCA_PhysPrior_wave1.csv")
tObs2 <- as.vector(t(tObs2)) ############## no chnage in obs in waves

# %%
cands <- names(tData)[1:4]
tLOOs <- LOO.plot(Emulators = TestEm2, which.emulator = 1, ParamNames = cands, Obs = tObs2[1], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm2, which.emulator = 2, ParamNames = cands, Obs = tObs2[2], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm2, which.emulator = 3, ParamNames = cands, Obs = tObs2[3], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm2, which.emulator = 4, ParamNames = cands, Obs = tObs2[4], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm2, which.emulator = 5, ParamNames = cands, Obs = tObs2[5], ObsErr = 0.)#tLOOs <- 

# %%
Timps2 <- matrix(rep(t(Timps),1), ncol=ncol(Timps), byrow=TRUE)
system.time(Timps2[NROY1,] <- ImplausibilityMOGP(NewData=Xp[NROY1,], Emulator=TestEm2, Discrepancy=tDisc, Obs=tObs2, ObsErr=tObsErr))

# %%
#tracemem(Timps2) == tracemem(Timps)

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
length(NROY2)

# %% [markdown]
# # Wave 3

# %%
40/ratio2

# %%
set.seed(42)

designpoints2 <- data.frame()

while (nrow(designpoints2) <= 40) { 
        ### Emulator wave 1
        tmp <- as.data.frame(2*maximinLHS(ceil(40/ratio2), 4)-1)
        names(tmp) <- names(TestEm$fitting.elements$Design)
        imps_tmp <- ImplausibilityMOGP(NewData=tmp, Emulator=TestEm, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp <- which(rowSums(imps_tmp <= cutoff_vec[1]) >= TestEm$mogp$n_emulators -valmax)
        ### Emulator wave 2
        imps_tmp2 <- ImplausibilityMOGP(NewData=tmp[NROYtmp,], Emulator=TestEm2, Discrepancy=tDisc, Obs=tObs2, ObsErr=tObsErr)
        NROYtmp2 <- which(rowSums(imps_tmp2 <= cutoff_vec[1]) >= TestEm2$mogp$n_emulators -valmax)
        #####
        selectionP <- tmp[NROYtmp,][NROYtmp2,]
        row.names(selectionP) <- NULL ## to avoid double index   
        designpoints2 <- rbind(designpoints2,selectionP)
        print(nrow(designpoints2))
        flush.console()
        } 

designpoints2 <- designpoints2[sample(nrow(designpoints2),40),]

designpoints_denorm2 <- rangeUnscale(designpoints2, my_bounds)

# %%
designpoints_denorm2

# %%
write.csv(designpoints_denorm2,"Data/exp_TuningL94_newPCA_PhysPrior_wave3.csv", row.names = FALSE)

# %%
inputs <- designpoints2

#Load outputs and select variables you want to keep
outputs <- read.csv('Data/df_metrics_newPCA_PhysPrior_wave3.csv')
                         
set.seed(42)
N = nrow(inputs) #nb samples 
noise <- rnorm(N, 0, 0.5)
tData <- cbind(inputs, noise, outputs)
names(tData)[names(tData) == "noise"] <- "Noise"
                         
head(tData)

# %%
TestEm3 <- BuildNewEmulators(tData,
                            HowManyEmulators = length(outputs),
                            meanFun = "fitted",
                            #kernel = c("Matern52"),
                            additionalVariables = names(tData)[1:4])  #important to put this line

                            #Choices = lapply(1:length(outputs),
                            #                   function(k) choices.new),

# %%
cands <- names(tData)[1:4]
tLOOs <- LOO.plot(Emulators = TestEm3, which.emulator = 1, ParamNames = cands, Obs = tObs2[1], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm3, which.emulator = 2, ParamNames = cands, Obs = tObs2[2], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm3, which.emulator = 3, ParamNames = cands, Obs = tObs2[3], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm3, which.emulator = 4, ParamNames = cands, Obs = tObs2[4], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm3, which.emulator = 5, ParamNames = cands, Obs = tObs2[5], ObsErr = 0.)#tLOOs <- 

# %%
Timps3 <- matrix(rep(t(Timps2),1), ncol=ncol(Timps2), byrow=TRUE)
system.time(Timps3[NROY2,] <- ImplausibilityMOGP(NewData=Xp[NROY2,], Emulator=TestEm3, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr))

# %%
ImpData_wave3 = cbind(Xp, Timps3)

# %%
valmax3 = 0 #how many outputs can be above the implausibility cut off?
ImpListM3 = CreateImpList(whichVars = 1:nparam, VarNames=VarNames, ImpData=ImpData_wave3, nEms=TestEm3$mogp$n_emulators, whichMax=valmax3+1)
NROY3 <- which(rowSums(Timps3 <= cutoff_vec[1]) >= TestEm3$mogp$n_emulators -valmax3)
ratio3 <- length(NROY3)/dim(Xp)[1]
ratio3

# %%
imp.layoutm11(ImpListM3,VarNames,VariableDensity=FALSE,newPDF=FALSE,the.title=paste("InputSpace_wave",WAVEN,".pdf",sep=""),newPNG=FALSE,newJPEG=FALSE,newEPS=FALSE,Points=matrix(param.defaults.norm,ncol=nparam))
mtext(paste("Remaining space:",length(NROY3)/dim(Xp)[1],sep=""), side=1)

# %%
length(NROY3)

# %% [markdown]
# ## wave 4

# %%
40/ratio3

# %% [markdown]
# switch to randomLHS

# %%
set.seed(42)

designpoints3 <- data.frame()

while (nrow(designpoints3) <= 40) { 
        ### Emulator wave 1
        tmp <- as.data.frame(2*randomLHS(ceil(40/ratio3), 4)-1)
        names(tmp) <- names(TestEm$fitting.elements$Design)
        imps_tmp <- ImplausibilityMOGP(NewData=tmp, Emulator=TestEm, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp <- which(rowSums(imps_tmp <= cutoff_vec[1]) >= TestEm$mogp$n_emulators -valmax)
        ### Emulator wave 2
        imps_tmp2 <- ImplausibilityMOGP(NewData=tmp[NROYtmp,], Emulator=TestEm2, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp2 <- which(rowSums(imps_tmp2 <= cutoff_vec[1]) >= TestEm2$mogp$n_emulators -valmax)
        ### Emulator wave 3
        imps_tmp3 <- ImplausibilityMOGP(NewData=tmp[NROYtmp,][NROYtmp2,], Emulator=TestEm3, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp3 <- which(rowSums(imps_tmp3 <= cutoff_vec[1]) >= TestEm3$mogp$n_emulators -valmax)
        #####
        selectionP <- tmp[NROYtmp,][NROYtmp2,][NROYtmp3,]
        row.names(selectionP) <- NULL ## to avoid double index   
        designpoints3 <- rbind(designpoints3,selectionP)
        print(nrow(designpoints3))
        flush.console()
        } 

designpoints3 <- designpoints3[sample(nrow(designpoints3),40),]

designpoints_denorm3 <- rangeUnscale(designpoints3, my_bounds)

# %%
write.csv(designpoints_denorm3,"Data/exp_TuningL94_newPCA_PhysPrior_wave4.csv", row.names = FALSE)

# %%
designpoints3

# %%
inputs <- designpoints3

#Load outputs and select variables you want to keep
outputs <- read.csv('Data/df_metrics_newPCA_PhysPrior_wave4.csv')
                         
set.seed(42)
N = nrow(inputs) #nb samples 
noise <- rnorm(N, 0, 0.5)
tData <- cbind(inputs, noise, outputs)
names(tData)[names(tData) == "noise"] <- "Noise"
                         
head(tData)

# %%
TestEm4 <- BuildNewEmulators(tData,
                            HowManyEmulators = length(outputs),
                            meanFun = "fitted",
                            #kernel = c("Matern52"),
                            additionalVariables = names(tData)[1:4])  #important to put this line

                            #Choices = lapply(1:length(outputs),
                            #                   function(k) choices.new),

# %%
cands <- names(tData)[1:4]
tLOOs <- LOO.plot(Emulators = TestEm4, which.emulator = 1, ParamNames = cands, Obs = tObs2[1], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm4, which.emulator = 2, ParamNames = cands, Obs = tObs2[2], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm4, which.emulator = 3, ParamNames = cands, Obs = tObs2[3], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm4, which.emulator = 4, ParamNames = cands, Obs = tObs2[4], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm4, which.emulator = 5, ParamNames = cands, Obs = tObs2[5], ObsErr = 0.)#tLOOs <- 

# %%
Timps4 <- matrix(rep(t(Timps3),1), ncol=ncol(Timps3), byrow=TRUE)
system.time(Timps4[NROY3,] <- ImplausibilityMOGP(NewData=Xp[NROY3,], Emulator=TestEm4, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr))

# %%
ImpData_wave4 = cbind(Xp, Timps4)

# %%
valmax4 = 0 #how many outputs can be above the implausibility cut off?
ImpListM4 = CreateImpList(whichVars = 1:nparam, VarNames=VarNames, ImpData=ImpData_wave4, nEms=TestEm4$mogp$n_emulators, whichMax=valmax4+1)
NROY4 <- which(rowSums(Timps4 <= cutoff_vec[1]) >= TestEm4$mogp$n_emulators -valmax4)
ratio4 <- length(NROY4)/dim(Xp)[1]
ratio4

# %%
imp.layoutm11(ImpListM4,VarNames,VariableDensity=FALSE,newPDF=FALSE,the.title=paste("InputSpace_wave",WAVEN,".pdf",sep=""),newPNG=FALSE,newJPEG=FALSE,newEPS=FALSE,Points=matrix(param.defaults.norm,ncol=nparam))
mtext(paste("Remaining space:",length(NROY4)/dim(Xp)[1],sep=""), side=1)

# %%
length(NROY4)

# %% [markdown]
# ## wave 5

# %%
40/ratio4 

# %%
set.seed(42)

designpoints4 <- data.frame()

while (nrow(designpoints4) <= 40) {
        ### Emulator wave 1
        tmp <- as.data.frame(2*randomLHS(ceil(40/ratio4), 4)-1)
        names(tmp) <- names(TestEm$fitting.elements$Design)
        imps_tmp <- ImplausibilityMOGP(NewData=tmp, Emulator=TestEm, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp <- which(rowSums(imps_tmp <= cutoff_vec[1]) >= TestEm$mogp$n_emulators -valmax)
        ### Emulator wave 2
        imps_tmp2 <- ImplausibilityMOGP(NewData=tmp[NROYtmp,], Emulator=TestEm2, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp2 <- which(rowSums(imps_tmp2 <= cutoff_vec[1]) >= TestEm2$mogp$n_emulators -valmax)
        ### Emulator wave 3
        imps_tmp3 <- ImplausibilityMOGP(NewData=tmp[NROYtmp,][NROYtmp2,], Emulator=TestEm3, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp3 <- which(rowSums(imps_tmp3 <= cutoff_vec[1]) >= TestEm3$mogp$n_emulators -valmax)
        ### Emulator wave 4
        imps_tmp4 <- ImplausibilityMOGP(NewData=tmp[NROYtmp,][NROYtmp2,][NROYtmp3,], Emulator=TestEm4, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
        NROYtmp4 <- which(rowSums(imps_tmp4 <= cutoff_vec[1]) >= TestEm4$mogp$n_emulators -valmax)
        #####
        selectionP <- tmp[NROYtmp,][NROYtmp2,][NROYtmp3,][NROYtmp4,]
        row.names(selectionP) <- NULL ## to avoid double index   
        designpoints4 <- rbind(designpoints4,selectionP)
        print(nrow(designpoints4))
        flush.console()
        } 

designpoints4 <- designpoints4[sample(nrow(designpoints4),40),]

designpoints_denorm4 <- rangeUnscale(designpoints4, my_bounds)

# %%
write.csv(designpoints_denorm4,"Data/exp_TuningL94_newPCA_PhysPrior_wave5.csv", row.names = FALSE)

# %%
inputs <- designpoints4

#Load outputs and select variables you want to keep
outputs <- read.csv('Data/df_metrics_newPCA_PhysPrior_wave5.csv')
                         
set.seed(42)
N = nrow(inputs) #nb samples 
noise <- rnorm(N, 0, 0.5)
tData <- cbind(inputs, noise, outputs)
names(tData)[names(tData) == "noise"] <- "Noise"
                         
head(tData)

# %%
TestEm5 <- BuildNewEmulators(tData,
                            HowManyEmulators = length(outputs),
                            meanFun = "fitted",
                            #kernel = c("Matern52"),
                            additionalVariables = names(tData)[1:4])  #important to put this line

                            #Choices = lapply(1:length(outputs),
                            #                   function(k) choices.new),

# %%
cands <- names(tData)[1:4]
tLOOs <- LOO.plot(Emulators = TestEm5, which.emulator = 1, ParamNames = cands, Obs = tObs2[1], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm5, which.emulator = 2, ParamNames = cands, Obs = tObs2[2], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm5, which.emulator = 3, ParamNames = cands, Obs = tObs2[3], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm5, which.emulator = 4, ParamNames = cands, Obs = tObs2[4], ObsErr = 0.)#tLOOs <- 
tLOOs <- LOO.plot(Emulators = TestEm5, which.emulator = 5, ParamNames = cands, Obs = tObs2[5], ObsErr = 0.)#tLOOs <- 

# %%
Timps5 <- matrix(rep(t(Timps4),1), ncol=ncol(Timps4), byrow=TRUE)
system.time(Timps5[NROY4,] <- ImplausibilityMOGP(NewData=Xp[NROY4,], Emulator=TestEm5, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr))

# %%
ImpData_wave5 = cbind(Xp, Timps5)

# %%
valmax5 = 0 #how many outputs can be above the implausibility cut off?
ImpListM5 = CreateImpList(whichVars = 1:nparam, VarNames=VarNames, ImpData=ImpData_wave5, nEms=TestEm5$mogp$n_emulators, whichMax=valmax4+1)
NROY5 <- which(rowSums(Timps5 <= cutoff_vec[1]) >= TestEm5$mogp$n_emulators -valmax5)
ratio5 <- length(NROY5)/dim(Xp)[1]
ratio5

# %%
png("InputSpace_PhysPrior_wave_5.png", res = 110)
imp.layoutm11(ImpListM5,VarNames,VariableDensity=FALSE,newPDF=FALSE,the.title=paste("InputSpace_wave",WAVEN,".pdf",sep=""),newPNG=FALSE,newJPEG=FALSE,newEPS=FALSE,Points=matrix(param.defaults.norm,ncol=nparam))
mtext(paste("Remaining space:",length(NROY5)/dim(Xp)[1],sep=""), side=1)
dev.off()

# %%
imp.layoutm11(ImpListM5,VarNames,VariableDensity=FALSE,newPDF=FALSE,the.title=paste("InputSpace_wave",WAVEN,".pdf",sep=""),newPNG=FALSE,newJPEG=FALSE,newEPS=FALSE,Points=matrix(param.defaults.norm,ncol=nparam))
mtext(paste("Remaining space:",length(NROY5)/dim(Xp)[1],sep=""), side=1)

# %%
length(NROY5)

# %% [markdown]
# ## K-means

# %%
library(ClusterR)

# %%
preProcValues <- preProcess(Xp[NROY5,], method = c("center", "scale"))
normalizeddata <- predict(preProcValues, Xp[NROY5,])

# %%
opt_km = Optimal_Clusters_KMeans(normalizeddata, 
                                 criterion = "silhouette", 
                                 max_clusters=10,
                                 plot_clusters = TRUE)

# %%
classif <- kmeans(normalizeddata, centers=3, iter.max=100, nstart=100)
kmcenters <- unPreProc(preProcValues, data.frame(classif$centers))
candidates <- rangeUnscale(kmcenters, my_bounds)
candidates

# %%
nrow(rangeUnscale(unPreProc(preProcValues, normalizeddata[classif$cluster==1,]), my_bounds))

# %%
library(xtable)

print.xtable(xtable(candidates))

# %% [markdown]
# # Ensemble of plausible simulations

# %%
### check if Kmeans centers are in NROY6
imps_kmeans <- ImplausibilityMOGP(NewData=kmcenters, Emulator=TestEm5, Discrepancy=tDisc, Obs=tObs, ObsErr=tObsErr)
which(rowSums(imps_kmeans <= cutoff_vec[1]) >= TestEm5$mogp$n_emulators -valmax5)

# %%
testpoints <- rangeUnscale(kmcenters, my_bounds)[rowSums(imps_kmeans <= cutoff_vec[1]) >= TestEm5$mogp$n_emulators -valmax5,]

# %%
write.csv(testpoints,"Data/finaltestpoints_newPCA_PhysPrior.csv", row.names = FALSE)

# %% [markdown]
# ## summary of the HM

# %%
NROYs <- 100 * c(ratio1, ratio2, ratio3, ratio4, ratio5)
NbSim <- c(40, nrow(designpoints), nrow(designpoints2), nrow(designpoints3), nrow(designpoints4))

# %%
data.frame(NROYs, NbSim)

# %%
print.xtable(xtable(data.frame(NROYs, NbSim)))

# %%
sum(NbSim)
