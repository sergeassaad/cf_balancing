#!/usr/bin/env Rscript
dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)  # create personal library
.libPaths(Sys.getenv("R_LIBS_USER"))  # add to the path
install.packages("optparse")
library(optparse)
library(reticulate)
library(grf)

option_list = list(
  make_option(c("-m", "--method"), type="character", default='MW'),
	make_option(c("-r", "--reps"), type="logical", default=FALSE),
  make_option(c("-n", "--normalized"), type="character", default='separately_normalized_weights')
)
 
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

print (paste('Method: ',opt$method))
print (paste('Using reps: ',opt$reps))
print (paste('Normalized: ',opt$normalized))

np <- import("numpy")

data_dir = './ihdp100_trees_neurips/'
method <- opt$method
use_reps <- opt$reps
normalized <- opt$normalized

npz.train <- np$load("./ihdp_npci_1-100.train.npz")
npz.test <- np$load("./ihdp_npci_1-100.test.npz")
ts = npz.train$f[['t']]

if (method == 'no_weight') {
  
  Xs = npz.train$f[['x']]
  Xs_te = npz.test$f[['x']]
  weights = NULL
  weights.test = NULL
  e_test = matrix(NA, dim(Xs)[1], dim(Xs)[3])
} else {
  
  npz.result.train <- np$load(paste(data_dir,method,'/results/result.npz', sep=''))
  npz.result.test <- np$load(paste(data_dir,method,'/results/result.test.npz', sep=''))
  # e_train = aperm(npz.result.train$f[['e']], c(2,1)) # no use
  e_test = aperm(npz.result.test$f[['e']], c(2,1))
  rm(npz.result.train, npz.result.test)
  
  npz.weight.train <- np$load(paste(data_dir,method,'/results/weights.npz', sep=''))
  npz.weight.test <- np$load(paste(data_dir,method,'/results/weights.test.npz', sep=''))
  weights = aperm(drop(npz.weight.train$f[['weights']]), c(2,1))
  # weights.test = aperm(drop(npz.weight.test$f[['weights']]), c(2,1)) # not used
  rm(npz.weight.train, npz.weight.test)

  if (use_reps) {
    
    npz.rep.train <- np$load(paste(data_dir,method,'/results/reps.npz', sep=''))
    npz.rep.test <- np$load(paste(data_dir,method,'/results/reps.test.npz', sep=''))

    Xs = aperm(drop(npz.rep.train$f[['rep']]), c(2,3,1))
    Xs_te = aperm(drop(npz.rep.test$f[['rep']]), c(2,3,1))

    rm(npz.rep.train, npz.rep.test)

  } else {
    Xs = npz.train$f[['x']]
    Xs_te = npz.test$f[['x']]
  }
  
  if (normalized == 'normalized_weights') {
    for (i in 1:dim(weights)[2]) {
      weights[,i] <- weights[,i] / sum(weights[,i])
    }
  } else if (normalized == 'separately_normalized_weights') {
    for (i in 1:dim(weights)[2]) {
      # print (sum(weights[,i]*ts[,i]))
      # print (sum(weights[,i]*(1-ts[,i])))
      # print (sum(weights[,i]))

      # print (ts[,i])
      # print (weights[,i])
      weights[,i] <- (ts[,i]*weights[,i] / sum(weights[,i]*ts[,i])) + ((1-ts[,i])*weights[,i] / sum(weights[,i]*(1-ts[,i])))
      # print (weights[,i])
      
      # print (sum(weights[,i]*ts[,i]))
      # print (sum(weights[,i]*(1-ts[,i])))
      # print (sum(weights[,i]))
      # quit()
    }
  }

}

Yfs = npz.train$f[['yf']]
mu0 = npz.train$f[['mu0']]
mu1 = npz.train$f[['mu1']]
tao.train.gt = mu1 - mu0
rm(mu0, mu1, npz.train)

ts_te = npz.test$f[['t']]
Yfs_te = npz.test$f[['yf']]
mu0 = npz.test$f[['mu0']]
mu1 = npz.test$f[['mu1']]
tao.test.gt = mu1 - mu0
rm(mu0, mu1, npz.test)

run_causal_forest <- function(X, Y, t, W) {
  
  # tau.forest <- causal_forest(X, Y, t, num.trees = 2000, tune.parameters='all', sampler.weights=W)
  if (!is.null(W)) { 
    tau.forest <- causal_forest(X, Y, t, num.trees = 2000, sample.weights=W, tune.parameters='all')
    #tau.forest <- causal_forest(X, Y, t, num.trees = 2000, sample.weights=W)
  } else {
    tau.forest <- causal_forest(X, Y, t, num.trees = 2000, tune.parameters='all')
  }

  # Estimate treatment effects for the training data using out-of-bag prediction.
  tau.hat.oob <- predict(tau.forest)
  # hist(tau.hat.oob$predictions)
  
  return (tau.forest)
}

test_causal_forest <- function(tau.forest, X_te) {
  # Estimate treatment effects for the test sample.
  tau.hat <- predict(tau.forest, X_te, estimate.variance = TRUE)
  return (tau.hat)
}

tau_ci_check <- function(val, mn, mx) {
  r = 0
  if (val <= mx && val >= mn) {
    r=1
  } 
  
  return (r)
}

tilting_function <- function(e, weight_scheme) {

  if (weight_scheme == 'no_weight' | weight_scheme == 'JW' | weight_scheme == 'IPW' | weight_scheme == 'ParetoIPW') {
    return (replicate(length(e), 1))
  } else if (weight_scheme == 'MW') {
    z = 1-e
    ind = which(e < (1-e))
    z[ind] = e[ind]
    return (z)
  } else if (weight_scheme == 'OW') {
    z <- e*(1-e)
    return (z)
  } else if (weight_scheme == 'TruncIPW') {
    z<-replicate(length(e), 1)
    ind<-which(0.1>e)
    z[ind]<-0
    ind<-which(e>0.9)
    z[ind]<-0
    return (z)
  } else {
    print ('Not a valid weighting scheme')
    quit()
  }
}

e_ATE.s <- numeric(dim(Xs)[3])
e_PEHE.s <- numeric(dim(Xs)[3])
tau_CI.s <- numeric(dim(Xs)[3])

e_ATE_g.s <- numeric(dim(Xs)[3])
e_PEHE_g.s <- numeric(dim(Xs)[3])

for (i in 1:dim(Xs)[3]) {
  
  X = Xs[,,i]
  t = ts[,i]
  Yf = Yfs[,i]  
  W = weights[,i]
  tau.forest <- run_causal_forest(X, Yf, t, W)
  
  X_te = Xs_te[,,i]
  
  tau.gt.value = tao.test.gt[,i]

  tau.pred <- test_causal_forest(tau.forest, X_te)
  tau.pred.value = tau.pred$predictions
  tau.pred.sigma <- sqrt(tau.pred$variance.estimates)
  
  ci_min <- tau.pred.value - 1.96*tau.pred.sigma
  ci_max <- tau.pred.value + 1.96*tau.pred.sigma

  m<-mapply(tau_ci_check, tau.gt.value, ci_min, ci_max)
  f<-tilting_function(e_test[,i] ,method)
  
  ate.pred.value = mean(tau.pred.value)
  ate.pred.g.value = sum(tau.pred.value * f)/sum(f)

  ate.gt.value = mean(tau.gt.value)
  ate.gt.g.value = sum(tau.gt.value*f)/sum(f)
  e_ATE = abs(ate.pred.value - ate.gt.value)
  e_ATE.g = abs(ate.pred.g.value - ate.gt.g.value)
  
  e_PEHE = sqrt(mean((tau.pred.value - tau.gt.value)^2))
  e_PEHE.g = sqrt(sum(f*(tau.pred.value - tau.gt.value)^2) / sum(f))
  
  e_ATE.s[i] = e_ATE
  e_PEHE.s[i] = e_PEHE
  tau_CI.s[i] = mean(m)
  e_ATE_g.s[i] = e_ATE.g
  e_PEHE_g.s[i] = e_PEHE.g
  
} 

# Results on IHDP100 Test set
avg_ePEHE = formatC(mean(e_PEHE.s), digits=4, format='f')
sdev_ePEHE = formatC(sd(e_PEHE.s)/sqrt(length(e_PEHE.s)), digits=4, format='f')
print ( paste('sqrt(e_PEHE): ', avg_ePEHE, ' +- ', sdev_ePEHE) )

avg_eATE = formatC(mean(e_ATE.s), digits=4, format='f')
sdev_eATE = formatC(sd(e_ATE.s)/sqrt(length(e_ATE.s)), digits=4, format='f')
print (paste('e_ATE: ', avg_eATE, ' +- ', sdev_eATE) )

avg_tau_CI = formatC(mean(tau_CI.s), digits=4, format='f')
sdev_tau_CI = formatC(sd(tau_CI.s)/sqrt(length(tau_CI.s)), digits=4, format='f')
print (paste('TAU_CI: ', avg_tau_CI, ' +- ', sdev_tau_CI))

avg_ePEHE.g = formatC(mean(e_PEHE_g.s), digits=4, format='f')
sdev_ePEHE.g = formatC(sd(e_PEHE_g.s)/sqrt(length(e_PEHE_g.s)), digits=4, format='f')
print ( paste('sqrt(e_PEHE.g): ', avg_ePEHE.g, ' +- ', sdev_ePEHE.g) )

avg_eATE.g = formatC(mean(e_ATE_g.s), digits=4, format='f')
sdev_eATE.g = formatC(sd(e_ATE_g.s)/sqrt(length(e_ATE_g.s)), digits=4, format='f')
print (paste('e_ATE.g: ', avg_eATE.g, ' +- ', sdev_eATE.g) )
