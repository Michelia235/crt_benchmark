# R implementation from https://github.com/moleibobliu/Distillation-CRT/
library('glmnet')
require(doMC)  # for parallelism

############# Auxillary functions #############

Creat_condition_gaussian <- function(X, indx, Sigma = NULL, lambda = NULL){
  if (is.null(Sigma)){
    cv_lasso_x <- cv.glmnet(X[,-indx], X[,indx], alpha = 1,
                            intercept = T, dfmax = as.integer(p / 2))
    lambda_x <- cv_lasso_x$lambda.min
    opt_model_x <- glmnet(X[,-indx], X[,indx], alpha = 1, 
                          lambda = lambda_x, intercept = T, dfmax = as.integer(p / 2))
    beta_x <- opt_model_x$beta
    X_bar <- predict(opt_model_x, X[,-indx])
    x_res <- X[,indx] - X_bar 
    sigma2_x <- mean(x_res^2)
  }else{
    beta_x <- solve(Sigma[-indx, -indx], Sigma[-indx, indx])
    X_bar <- X[ ,-indx] %*% beta_x
    sigma2_x <- Sigma[indx, indx] - Sigma[indx, -indx] %*% beta_x
    ##print(beta_x)
  }
  return(list(mean_x = X_bar, sigma2_x = sigma2_x, gamma = beta_x))
}


AR_cov <- function(p, ar){
  ar_series <- ar^(c(1:p) - 1)
  cov_mat <- ar_series
  for (i in 1:(p - 1)){
    cov_mat <- cbind(cov_mat, ar_series[c((p - i + 1):p ,1:(p - i))])
  }
  for (i in 1:(p - 1)){
    for (j in (i + 1):p){
      cov_mat[i, j] <- cov_mat[j, i]
    }
  }
  rownames(cov_mat) <- colnames(cov_mat)
  return(cov_mat)
}


cons_cov <- function(p, r){
  mat <- diag(1, p)
  for (i in 1:p){
    for (j in 1:p){
      if (i != j){
        mat[i,j] <- r
      }
      
    }
  }
  return(mat)
}


################## High dim refit CRT  ##################

smc_est <- function(Y, X, indx, X_bar, sigma2_x, 
                    lamb, beta_ob, type = 'LASSO', model = 'gaussian'){
  N <- length(Y)
  p <- length(X[1, ])
  
  delta_gen <- rnorm(N, 0, sqrt(sigma2_x))
  X_sample <- X_bar + delta_gen
  
  # Fit the regression
  
  X[ ,indx] <- X_sample
  
  if (type == 'LASSO'){
    opt_model <- glmnet(X, Y, alpha = 1, lambda = lamb, intercept = T, family = model)
    beta_fit <- opt_model$beta
    t <- beta_fit[indx]
  }
  if (type == 'Elasnet'){
    opt_model <- glmnet(X, Y, alpha = 1 / 3, lambda = lamb, intercept = T, family = model)
    beta_fit <- opt_model$beta
    t <- beta_fit[indx]
  }
  if (type == 'AdaLASSO'){
    adalasso <- adapt_lasso(X, Y, CV = F, lamb_lasso = lamb, model = model)
    beta_fit <- adalasso$beta
    t <- beta_fit[indx]
  }
  
  f <- ifelse(abs(t) >= abs(beta_ob), 1, 0)
  return(f)
}


adapt_lasso <- function(X, Y, CV = T, lamb_lasso = 1, model = 'gaussian'){
  
  if (CV == T){
    cv.lasso <- cv.glmnet(X, Y, alpha = 1, intercept = T, family = model)
    lamb_lasso <- cv.lasso$lambda.min
  }
  
  lasso_model_1 <- glmnet(X, Y, alpha = 1, 
                          lambda = lamb_lasso, intercept = T, family = model)
  S <- which(lasso_model_1$beta != 0)
  
  weight <- 1 / abs(lasso_model_1$beta[S])
  X_S <- X[,S]
  
  opt_model <- glmnet(X_S, Y, alpha = 1, lambda = lamb_lasso, 
                      penalty.factor = weight, intercept = T, family = model)
  beta_coef <- rep(0, length(X[1,]))
  beta_coef[S] <- opt_model$beta
  
  return(list(lamb_lasso = lamb_lasso, beta = beta_coef))
}



CRT_sMC <- function(Y, X, Sigma_X, m = 1000, set_use = 1:ncol(X),
                    type = 'LASSO', FDR = 0.1, model = 'gaussian', n_jobs = 1){
  p <- length(X[1, ])
  n <- length(Y)
  
  registerDoMC(cores = 2 * n_jobs)  # parallelism
  
  if (type == 'LASSO'){
    cv_lasso <- cv.glmnet(X, Y, alpha = 1, intercept = T, parallel = T,
                          family = model)
    lamb <- cv_lasso$lambda.min
    opt_model <- glmnet(X, Y, alpha = 1, lambda = lamb, 
                        intercept = T, family = model)
    beta_fit <- opt_model$beta
  }
  if (type == 'Elasnet'){
    cv_ela <- cv.glmnet(X, Y, alpha = 1 / 3, intercept = T, family = model)
    lamb <- cv_ela$lambda.min
    opt_model <- glmnet(X, Y, alpha = 1 / 3, lambda = lamb, 
                        intercept = T, family = model)
    beta_fit <- opt_model$beta
  }
  if (type == 'AdaLASSO'){
    adalasso <- adapt_lasso(X, Y, CV = T, model = model)
    lamb <- adalasso$lamb_lasso
    beta_fit <- adalasso$beta
  }
  
  nonzero_set <- which(beta_fit != 0)
  
  nonzero_set <- intersect(nonzero_set, set_use)
  if (is.null(nonzero_set)){
    return(c())
  }
  
  s_0 <- length(nonzero_set)
  
  X_bar_lst <- matrix(0, n, s_0)
  sigma2_x_lst <- rep(0, s_0)
  
  # Construct Statistics and IS
  
  for (j in 1:s_0){
    indx <- nonzero_set[j]
    Cond_X <- Creat_condition_gaussian(X, indx, Sigma = Sigma_X)
    X_bar_lst[,j] <- Cond_X$mean_x
    sigma2_x_lst[j] <- Cond_X$sigma2_x
  }
  
  # Monte Carlo 
  
  pvl_lst <- rep(1, p)  
  
  for (t in 1:s_0){
    indx <- nonzero_set[t]
    Z_vec <- c(1)
    for (i in 1:m){
      Z <- smc_est(Y, X, indx, X_bar_lst[,t], sigma2_x_lst[t], 
                   lamb, beta_fit[indx], type = type, model = model)
      Z_vec <- c(Z_vec, Z)
    }
    pvl_lst[indx] <- mean(Z_vec)
  }
  CR_BHq_lst <- p.adjust(pvl_lst, method = 'BH')
  
  selection_set_CR <- which(CR_BHq_lst <= FDR)
  return(list(select_set = selection_set_CR, pvl = pvl_lst))
}



CRT_modified <- function(Y, X, Sigma_X, m = 2000, model = 'gaussian', candidate_set = 1){
  p <- length(X[1, ])
  n <- length(Y)
  
  cv_lasso <- cv.glmnet(X, Y, alpha = 1, intercept = T, family = model)
  lamb <- cv_lasso$lambda.min
  opt_model <- glmnet(X, Y, alpha = 1, lambda = lamb, intercept = T, family = model)
  beta_fit <- opt_model$beta
  
  j <- candidate_set
  Cond_X <- Creat_condition_gaussian(X, j, Sigma = Sigma_X)
  X_bar <- Cond_X$mean_x
  sigma2 <- Cond_X$sigma2_x
  
  test_stat <- c(0, 0, 0, 0)
  
  for (type in c('origin', 'distill_1', 'distill_2', 'sq_loss')){
    if (type == 'origin'){
      test_stat[1] <- beta_fit[j]
    }
    if (type == 'distill_1'){
      offsets <- opt_model$a0 + X[,-j] %*% beta_fit[-j]
      if (model == 'binomial'){
        Y_res  <- Y - 1 / (1 + exp(- offsets))
      }
      if (model == 'gaussian'){
        Y_res  <- Y - offsets
      }
      X_res <- X[, j]
      test_stat[2] <- mean(X_res * Y_res)
      
    }
    if (type == 'distill_2'){
      
      offsets <- opt_model$a0 + X[,-j] %*% beta_fit[-j] + X_bar * beta_fit[j]
      
      if (model == 'binomial'){
        Y_res  <- Y - 1 / (1 + exp(- offsets))
      }
      if (model == 'gaussian'){
        Y_res  <- Y - offsets
      }
      X_res <- X[, j] - X_bar
      test_stat[3] <- mean(X_res * Y_res)
    }
    if (type == 'sq_loss'){
      test_stat[4] <- mean(Y^2) - mean((Y - X %*% beta_fit - opt_model$a0)^2)
    }
  }
  
  reject_list <- vector('list', 4)
  for (t in 1:4){
    reject_list[[t]] <- c(1)
  }
  
  coef_mat <- opt_model$beta
  test_mat <- test_stat
  for (i in 1:m){
    test_resample <- smc_est_new(Y, X, j, X_bar, sigma2, lamb, model = model)
    for (t in 1:4){
      if (abs(test_resample$test_stat[t]) >= abs(test_stat)[t]){
        reject_list[[t]] <- c(reject_list[[t]], 1)
      }else{
        reject_list[[t]] <- c(reject_list[[t]], 0)
      }
    }
    coef_mat <- cbind(coef_mat, as.vector(test_resample$coef))
    test_mat <- cbind(test_mat, as.vector(test_resample$test_stat))
  }
  p_value = rep(1, 6)
  for (t in 1:4){
    p_value[t] <- mean(reject_list[[t]])
  }
  stats <- test_mat[2, ]
  p_value[5] <- 2 * min(mean(ifelse(stats >= stats[1], 1, 0)),
                        mean(ifelse(stats <= stats[1], 1, 0)))
  stats <- abs(stats - mean(stats))
  p_value[6] <- mean(ifelse(stats >= stats[1], 1, 0))
  
  return(list(p_value = p_value, cv.lambda = lamb, coef = coef_mat, test_stat = test_mat))
}




smc_est_new <- function(Y, X, j, X_bar, sigma2_x, lamb, model = 'gaussian'){
  N <- length(Y)
  p <- length(X[1, ])
  
  delta_gen <- rnorm(N, 0, sqrt(sigma2_x))
  X_sample <- X_bar + delta_gen
  
  # Fit the regression
  
  X[ ,j] <- X_sample
  
  opt_model <- glmnet(X, Y, alpha = 1, lambda = lamb, intercept = T, family = model)
  beta_fit <- opt_model$beta
  test_stat <- c(0, 0, 0, 0)
  
  for (type in c('origin', 'distill_1', 'distill_2', 'sq_loss')) {
    if (type == 'origin'){
      test_stat[1] <- beta_fit[j]
    }
    if (type == 'distill_1'){
      
      offsets <- opt_model$a0 + X[,-j] %*% beta_fit[-j]
      
      if (model == 'binomial'){
        Y_res  <- Y - 1 / (1 + exp(- offsets))
      }
      if (model == 'gaussian'){
        Y_res  <- Y - offsets
      }
      X_res <- X[, j]
      test_stat[2] <- mean(X_res * Y_res)
    }
    if (type == 'distill_2'){
      
      offsets <- opt_model$a0 + X[,-j] %*% beta_fit[-j] + X_bar * beta_fit[j]
      
      if (model == 'binomial'){
        Y_res  <- Y - 1 / (1 + exp(- offsets))
      }
      if (model == 'gaussian'){
        Y_res  <- Y - offsets
      }
      X_res <- X[, j] - X_bar
      test_stat[3] <- mean(X_res * Y_res)
    }
    if (type == 'sq_loss'){
      test_stat[4] <- mean(Y^2) - mean((Y - X %*% beta_fit - opt_model$a0)^2)
    }
  }
  
  return(list(coef = opt_model$beta, test_stat = test_stat))
}



################## DML ########################

DML <- function(X, Y, Sigma_X, FDR = 0.1, K = 5, model = 'gaussian', 
                test_set = 1:length(X[1,])){
  
  p <- length(X[1, ])
  n <- length(Y)
  pvl_lst <- rep(1, p)
  
  for (indx in test_set){
    
    Cond_X <- Creat_condition_gaussian(X, indx, Sigma = Sigma_X)
    X_bar <- Cond_X$mean_x
    sigma2_x <- Cond_X$sigma2_x
    cv_lasso <- cv.glmnet(X[,-indx], Y, alpha = 1, dfmax = as.integer(p / 2))
    lamb <- cv_lasso$lambda.min
    eps_res <- c()
    
    for (k in 1:K) {
      split_set <- c(((k - 1) * as.integer(n / K) + 1):(k * as.integer(n / K)))
      model_res_null <- glmnet(X[-split_set,-indx], Y[-split_set], alpha = 1, 
                               lambda = lamb, family = model, dfmax = as.integer(p / 2))
      eps_res <- c(eps_res, Y[split_set] - 
                     predict(model_res_null, X[split_set,-indx], type = 'response'))
      if (k == K & as.integer(n / K) < n / K){
        remain_set <- c((k * as.integer(n / K) + 1):n)
        eps_res <- c(eps_res, Y[remain_set] - 
                       predict(model_res_null, X[remain_set,-indx], type = 'response'))
      }
    }
    
    X_res <- X[,indx] - X_bar
    r <- mean(eps_res * X_res)
    emp_var <- mean(eps_res^2) * sigma2_x
    pvl <- 1 - pnorm(sqrt(n) * abs(r) / sqrt(emp_var)) + pnorm(- sqrt(n) * abs(r) / sqrt(emp_var))
    
    pvl_lst[indx] <- pvl
    print('################## DML pvalue ##################')
    print(paste(indx, pvl, sep = ': '))
  }
  CR_BHq_lst <- p.adjust(pvl_lst, method = 'BH')
  selection_set_CR <- which(CR_BHq_lst <= FDR)
  return(list(select_set = selection_set_CR, p_values = pvl_lst))
}



################## GCM ########################

GCM <- function(X, Y, Sigma_X, FDR = 0.1, model = 'gaussian', 
                test_set = 1:length(X[1,])){
  
  p <- length(X[1, ])
  n <- length(Y)
  pvl_lst <- rep(1, p)
  
  for (indx in test_set){
    
    Cond_X <- Creat_condition_gaussian(X, indx, Sigma = Sigma_X)
    X_bar <- Cond_X$mean_x
    sigma2_x <- Cond_X$sigma2_x
    cv_lasso <- cv.glmnet(X[,-indx], Y, alpha = 1, dfmax = as.integer(p / 2))
    lamb <- cv_lasso$lambda.min

    model_res_null <- glmnet(X[,-indx], Y, alpha = 1, 
                             lambda = lamb, family = model, dfmax = as.integer(p / 2))
    eps_res <- Y - predict(model_res_null, X[,-indx], type = 'response')
    
    X_res <- X[,indx] - X_bar
    r <- mean(eps_res * X_res)
    emp_var <- mean(eps_res^2) * sigma2_x
    pvl <- 1 - pnorm(sqrt(n) * abs(r) / sqrt(emp_var)) + pnorm(- sqrt(n) * abs(r) / sqrt(emp_var))
    
    pvl_lst[indx] <- pvl
    print('################## GCM pvalue ##################')
    print(paste(indx, pvl, sep = ': '))
  }
  CR_BHq_lst <- p.adjust(pvl_lst, method = 'BH')
  selection_set_CR <- which(CR_BHq_lst <= FDR)
  return(list(select_set = selection_set_CR, p_values = pvl_lst))
}


######################### HRT #########################


HRT <- function(Y, X, Sigma = NULL, FDR = 0.1, N = 30000,
                model_select = 'CV', 
                lamb = 2 * log(length(X[1, ])) / length(train_set),
                pvl_study = F, S = NULL, X_bar_mat = 0, x_type = 'gauss', 
                model = 'gaussian', test_group = 1:p, single = F, n_jobs = 1){
    
  registerDoMC(cores = 2 * n_jobs)  # parallelism  
  p <- length(X[1,])
  ##l_diff_total <- matrix(0, N, p)
  n <- length(Y)
  num <- as.integer(n / 2)
  pvl_lst <- rep(1, p)
  
  test_set <- c(1:num)
  train_set <- setdiff(1:n, test_set)
  Y_train <- Y[train_set]
  X_train <- X[train_set,]
  X_test <- X[test_set,]
  Y_test <- Y[test_set]
  
  
  if (model_select == 'CV'){
    cv_lasso <- cv.glmnet(X_train, Y_train, alpha = 1, parallel = T,
                          intercept = T, family = model, dfmax = as.integer(p / 2))
    lamb <- cv_lasso$lambda.min
    opt_model <- glmnet(X_train, Y_train, alpha = 1, lambda = lamb,
                        intercept = T, family = model, dfmax = as.integer(p / 2))
  }
  if (model_select == 'fix'){
    opt_model <- glmnet(X_train, Y_train, alpha = 1, lambda = lamb, 
                        intercept = T, family = model, dfmax = as.integer(p / 2))
  }
  
  beta_fit <- opt_model$beta
  nonzero_set <- which(beta_fit != 0)
  
  if (pvl_study == T){
    nonzero_set <- setdiff(nonzero_set, S)
  }
  nonzero_set <- intersect(nonzero_set, test_group)
  if (model == 'gaussian'){
    l_ob <- mean((Y[test_set] - predict(opt_model, X[test_set,]))^2)
  }
  if (model == 'binomial'){
    l_ob <- - mean(Y[test_set] * log(predict(opt_model, 
                                             X[test_set,], type = 'response')) + 
                     (1 - Y[test_set]) * log(1 - predict(opt_model, 
                                                         X[test_set,], type = 'response'))) 
  }
  
  if (is.null(nonzero_set)){
    return(list(select_set = c()))
  }
  
  
  if (single == T){
    nonzero_set <- c(1)
  }
  
  for (j in nonzero_set){
    indx <- j
    
    if (x_type == 'gauss'){
      Cond_X <- Creat_condition_gaussian(X, indx, Sigma = Sigma)
      X_bar <- Cond_X$mean_x
      sigma2_x <- Cond_X$sigma2_x
    }
    
    l_diff_lst <- rep(0, N)
    if (beta_fit[j] == 0){
      
    }else{
      if (x_type == 'gauss'){
        l_sample_lst <- HRT_j(Y_test, X_test, 
                              Sigma = Sigma, opt_model, indx,
                              X_bar, sigma2_x, test_set, N = N, 
                              x_type = x_type, model = model)
      }
      if (x_type == 'indept_binary'){
        l_sample_lst <- HRT_j(Y_test, X_test, 
                              Sigma = 0, opt_model, indx,
                              X_bar = 0,sigma2_x = 0, test_set, N = N, 
                              x_type = x_type, model = model)
      }
      if (x_type == 'laplace'){
        l_sample_lst <- HRT_j(Y_test, X_test, 
                              Sigma = 0, opt_model, indx,
                              X_bar = 0,sigma2_x = 0, test_set, N = N, 
                              x_type = x_type, model = model)
      }
      if (x_type == 'indept_gamma'){
        l_sample_lst <- HRT_j(Y_test, X_test, 
                              Sigma = 0, opt_model, indx,
                              X_bar = 0, sigma2_x = 0, test_set, N = N, 
                              x_type = x_type, model = model)
      }
      l_diff_lst <- l_diff_lst + l_ob - l_sample_lst
      
    }
    pvl_lst[j] <- (length(which(l_diff_lst >= 0)) + 1) / (N + 1)
  }
  
  if (x_type == 'indept_binary'){
    pvl_lst[setdiff(1:p, test_group)] <- runif(length(setdiff(1:p, test_group)), 0, 1)
  }
  if (x_type == 'indept_gamma'){
    pvl_lst[setdiff(1:p, test_group)] <- runif(length(setdiff(1:p, test_group)), 0, 1)
  }
  
  set_select <- which(pvl_lst != 1)
  pvl_select <- pvl_lst[set_select]
  CRT_BHq_lst <- p.adjust(pvl_select, method = 'BH')
  select_BHq <- which(CRT_BHq_lst <= FDR)
  select_BHq <- set_select[select_BHq]
  
  select_Bf <- which(pvl_select <= min(0.1 / length(set_select), 0.1))
  select_Bf <- set_select[select_Bf]
    
  #CR_BHq_lst <- p.adjust(pvl_lst, method = 'BH')
  #selection_set_CR <- which(CR_BHq_lst <= FDR)
  return(list(select_set = select_BHq, p_values = pvl_lst, select_FWER = select_Bf))
  
}

HRT_j <- function(Y_test, X_test, Sigma = NULL, opt_model, indx,
                  X_bar, sigma2_x, test_set, N = 30000,
                  x_type = 'gauss', model = 'gaussian'){
  l_sample_lst <- c()
  n_test <- length(Y_test)
  
  if (x_type == 'gauss'){
    X_sample_all <- rnorm(n_test * N, 0, sqrt(sigma2_x))
    X_sample_all <- matrix(X_sample_all, n_test, N) + X_bar[test_set]
  }
  if (x_type == 'indept_binary'){
    X_sample_all <- rbinom(n_test * N, 1, 0.5)
    X_sample_all <- matrix(X_sample_all, n_test, N)
    X_sample_all <- (X_sample_all - 0.5) * 2
  }
  if (x_type == 'indept_gamma'){
    X_sample_all <- rgamma(n_test * N, shape = 3, rate = 0.5)
    X_sample_all <- matrix(X_sample_all, n_test, N)
    X_sample_all <- (X_sample_all - 6) / sqrt(12)
  }
  
  if (x_type == 'laplace'){
    sign_X <- 2 * rbinom(n_test * N, 1, 0.5) - 1
    X_sample_all <- sign_X * rexp(n_test * N, 3) 
    X_sample_all <- matrix(X_sample_all, n_test, N)
  }
  
  X_test_null <- X_test
  X_test_null[,indx] <- 0
  offsets <- predict(opt_model, X_test_null, type = 'response')
  if (model == 'binomial'){
    offsets <- log(offsets / (1 - offsets))
  }
  pred.value <- as.vector(offsets) + opt_model$beta[indx] * X_sample_all
  if (model == 'gaussian'){
    l_sample_lst <- colMeans((Y_test - pred.value)^2) 
  }
  if (model == 'binomial'){
    pred.value <- 1 / (1 + exp(- pred.value))
    l_sample_lst <- - colMeans(Y_test * log(pred.value) + (1 - Y_test) * log(1 - pred.value))
  }
  return(l_sample_lst)
}





######################### Estimate covariance matrix #########################


### Ledoit–Wolf (optimal shrinkage) estimator: 

linshrink_cov <- function(X, k = 0, normalize = F) 
{
  n <- nrow(X)
  p <- ncol(X)
  if (k == 0) {
    X <- X - tcrossprod(rep(1, n), colMeans(X))
    k = 1
  }
  if (n > k) 
    effn <- n - k
  else stop("k must be strictly less than nrow(X)")
  S <- crossprod(X)/effn
  Ip <- diag(p)
  m <- sum(S * Ip)/p
  d2 <- sum((S - m * Ip)^2)/p
  b_bar2 <- 1/(p * effn^2) * sum(apply(X, 1, function(x) sum((tcrossprod(x) - 
                                                                S)^2)))
  b2 <- min(d2, b_bar2)
  a2 <- d2 - b2
  Sigma_true <- b2/d2 * m * Ip + a2/d2 * S
  
  if (normalize == T){
    var_lst <- c()
    for (indx in 1:p){
      beta_x <- solve(Sigma_true[-indx, -indx], Sigma_true[-indx, indx])
      X_bar <- X[ ,-indx] %*% beta_x
      var_lst <- c(var_lst, mean((X[,indx] - X_bar)^2))
    }
    theta_lst <- diag(solve(Sigma_true))
    d <- sqrt(theta_lst * var_lst)
    Sigma_true <- as.matrix(diag(d)) %*% as.matrix(Sigma_true) %*% as.matrix(diag(d))
  }
  
  return(Sigma_true)
}


### Graphic lasso estimator:

glasso_cov <- function(X, prop = 0){
  prec_est <- CVglasso(X = X, lam.min.ratio = 3e-2)
  length(which(prec_est$Omega != 0))
  Sigma_true <- prec_est$Sigma * (1 - prop) + cov(X) * prop
  
  var_lst <- c()
  for (indx in 1:p){
    beta_x <- solve(Sigma_true[-indx, -indx], Sigma_true[-indx, indx])
    X_bar <- X[ ,-indx] %*% beta_x
    var_lst <- c(var_lst, mean((X[,indx] - X_bar)^2))
    print(indx)
  }
  theta_lst <- diag(solve(Sigma_true))
  d <- sqrt(theta_lst * var_lst)
  
  return(as.matrix(diag(d)) %*% as.matrix(Sigma_true) %*% as.matrix(diag(d)))
  
}





################## Generate data for simulation ########################

Generate_data <- function(N, p, s = 20, intercept = 0, model = 'linear', Y_dist = 'Gaussian',
                          Sigma = 'AR', r = 0.4, magn = 0.5, X_dist = 'gauss', para_x = 0.5, prop = 0,
                          support = 'first', sign_design = 'random', support_dist = NULL){
  
  if (sign_design == 'random'){
    sign_beta <- 2 * rbinom(s, 1, 0.5) - 1
  }
  
  if (sign_design == 'pos'){
    sign_beta <- rep(1, s)
  }
  
  if (sign_design == 'neg'){
    sign_beta <- c(1, rep(-1, s - 1))
  }
  
  if (sign_design == 'half'){
    sign_beta <- rep(c(1, -1), s / 2)
    if (s / 2 != as.integer(s / 2)){
      print('s needs to be an even number.')
    }
  }
  
  
  if (is.null(support_dist)){
    if (support == 'first'){
      beta_coef <-  c(sign_beta * rep(magn, s), rep(0, p - s))
    }
    if (support == 'random'){
      beta_coef <- rep(0, p)
      beta_coef[sample(1:p, s)] <- sign_beta * rep(magn, s)
    }
    
    if (support == 'equal'){
      beta_coef <- rep(0, p)
      beta_coef[1 + 0:(s - 1) * (as.integer(p / s))] <- sign_beta * rep(magn, s)
    }
  }else{
    beta_coef <- rep(0, p)
    beta_coef[seq(1, s * support_dist, support_dist)] <- sign_beta * magn
  }
  
  
  if (X_dist == 'gauss'){
    if (Sigma == 'AR'){
      Sigma_Design <- AR_cov(p, r)
      X <- mvrnorm(N, rep(0, p), Sigma_Design)
    }
    if (Sigma == 'Cons'){
      Sigma_Design <- cons_cov(p, r)
      X <- mvrnorm(N, rep(0, p), Sigma_Design)
    }
    
    if (Sigma == 'single'){
      X_rem <- mvrnorm(N, rep(0, p - s), diag(rep(1, p - s)))
      gamma_coef <- c()
      for (t in 1:s) {
        gamma_coef <- cbind(gamma_coef, r * rbinom(p - s, 1, prop))
      }
      
      X <- cbind(X_rem %*% gamma_coef + matrix(rnorm(s * N, 0, 1), N, s), X_rem)
      Sigma_Design <- rbind(cbind(diag(rep(1, s)) + t(gamma_coef) %*% gamma_coef, t(gamma_coef)), 
                            cbind(gamma_coef, diag(rep(1, p - s))))
      
    }
    Design <- X
  }
  if (X_dist == 'binom'){
    Sigma_latent <- AR_cov(p, r)
    X_latent <- mvrnorm(N, rep(0, p), Sigma_latent)
    X <- ifelse(X_latent > 0, 1, 0)
    Design <- X
    
    N_sim <- 30000
    X_sim_latent <- mvrnorm(N_sim, rep(0, p), Sigma_latent)
    X_sim <- ifelse(X_sim_latent > 0, 1, 0)
    Sigma_Design <- t(X_sim) %*% X_sim / N_sim
  }
  
  if (X_dist == 'gamma'){
    Sigma_latent <- AR_cov(p, r)
    X_latent <- mvrnorm(N, rep(0, p), Sigma_latent)
    
    X <- rgamma(N * p, shape = as.vector(exp(0.25 * X_latent)), rate = 1)
    X <- matrix(X, N, p)
    Design <- X
    
    N_sim <- 30000
    X_sim_latent <- mvrnorm(N_sim, rep(0, p), Sigma_latent)
    X_sim <- rgamma(N_sim * p, 
                    shape = as.vector(exp(0.25 * X_sim_latent)), rate = 1)
    X_sim <- matrix(X_sim, N_sim, p)
    Sigma_Design <- t(X_sim) %*% X_sim / N_sim
    
  }
  
  if (X_dist == 'poisson'){
    Sigma_Design <- para_x
    X <- rpois(N * p, Sigma_Design)
    X <- matrix(X, N, p)
    Design <- X
  }
  
  if (X_dist == 'poisson_single'){
    Sigma_remain <- AR_cov(p - 1, r)
    X_remain <- mvrnorm(N, rep(0, p - 1), Sigma_remain)
    
    sign_gamma <- 2 * rbinom(50, 1, 0.5) - 1
    gamma_coef <- rep(0, p - 1)
    gamma_coef[1:50] <- sign_gamma * rep(0.125, 50)
    
    X1 <- X_remain %*% gamma_coef + (rpois(N, para_x) - para_x) / sqrt(para_x)
    X <- cbind(X1, X_remain)
    Design <- X
    ttt <- Sigma_remain %*% gamma_coef
    Sigma_Design <- rbind(cbind(1 + t(ttt) %*% gamma_coef, t(ttt)), 
                          cbind(ttt, Sigma_remain))
  }
  
  if (X_dist == 'Laplace'){
    Sigma_Design <- para_x
    sign_X <- 2 * rbinom(N * p, 1, 0.5) - 1
    X <- sign_X * rexp(N * p, para_x) 
    X <- matrix(X, N, p)
    Design <- X
  }
  
  if (X_dist == 'indept_gamma'){
    X <- rgamma(N * p, shape = 3, rate = 0.5)
    X <- matrix(X, N, p)
    X <- (X - 6) / sqrt(12)
    Design <- X
    Sigma_Design <- 1
  }
  
  if (X_dist == 'indept_binary'){
    X <- rbinom(N * p, 1, 0.5)
    X <- matrix(X, N, p)
    X <- (X - 0.5) * 2
    Design <- X
    Sigma_Design <- 1
  }
  
  if (X_dist == 'indept_gaussian'){
    X <- rnorm(N * p, 0, 1)
    X <- matrix(X, N, p)
    Design <- X
    Sigma_Design <- 1
  }
  
  if (model == 'linear'){
    if (X_dist == 'poisson_single'){
      sign_beta <- 2 * rbinom(50, 1, 0.5) - 1
      beta_coef <- rep(0, p - 1)
      beta_coef[1:50] <- sign_beta * rep(0.125, 50)
      
      mean_y <- magn * Design[,1] + Design[,-1] %*% beta_coef
    }else{
      mean_y <- Design %*% beta_coef + intercept
    }
  }
  
  if (model == 'square'){
    mean_y <- Design^2 %*% beta_coef + intercept
  }
  
  if (model == 'poly'){
    mean_y <- intercept + Design %*% beta_coef + 
      0.3 * Design^3 %*% beta_coef 
  }
  if (model == 'exp'){
    mean_y <- exp(Design %*% beta_coef + intercept)
  }
  if (model == 'mix'){
    mean_0 <- Design %*% beta_coef + intercept
    mean_1 <- 0.5 * Design^2 %*% beta_coef + intercept
    cluster_y <- rbinom(N, 1, 0.5)
    mean_y <- mean_0 * cluster_y + mean_1 * (1 - cluster_y)
    var_y <- 1 * cluster_y + 4  * (1 - cluster_y)
  }
  
  if (model == 'non_linear'){
    support <- sample(1:p, s)
    beta_coef <- rep(0, p)
    beta_coef[support] <- 1
    interaction_mat <- c()
    for (i in 1:(s - 1)){
      for (j in (i + 1):s){
        interaction_mat <- cbind(interaction_mat,
                                 Design[,support[i]] * Design[,support[j]])
      }
    }
    mean_y <- magn * (rowSums(Design[,support]) +
                        1.5 * rowSums(interaction_mat)) + intercept
  }
  
  if (model == 'non_linear_single'){
    support <- sample(2:p, s)
    beta_coef <- rep(0, p)
    interaction_mat <- c()
    for (i in 1:s){
      interaction_mat <- cbind(interaction_mat,
                               Design[,1] * Design[,support[i]])
      
    }
    mean_y <- magn * (rowSums(Design[,support]) + Design[,1] +
                        2 * rowSums(interaction_mat))
  }
  
  sigma2 <- 1
  
  if (Y_dist == 'Gaussian'){
    Y <- rnorm(N, mean_y, sqrt(sigma2))
  }
  if (Y_dist == 'Poisson'){
    Y <- log(rpois(N, mean_y) + 1)
  }
  if (Y_dist == 'binom'){
    ##print(exp(mean_y) / (1 + exp(mean_y)))
    Y <- rbinom(N, 1, exp(mean_y) / (1 + exp(mean_y)))
  }
  
  if (Y_dist == 'Laplace'){
    sign_Y <- 2 * rbinom(N, 1, 0.5) - 1
    Y <- sign_Y * rexp(N, 2) + mean_y
  }  
  
  if (Y_dist == 'mix'){
    Y <- rnorm(N, mean_y, sqrt(var_y))
  }
  
  return(list(Y = Y, X = X, beta_true = beta_coef, Sigma = Sigma_Design))
}


Gen_forest <- function(N, p, s = 5, r = 0, intercept = 0, 
                       Sigma = 'AR', X_dist = 'gauss', prop = 0,
                       magn = 1, s_x = 10, Y_dist = 'Gaussian'){
  
  if (X_dist == 'gauss'){
    if (Sigma == 'AR'){
      Sigma_Design <- AR_cov(p, r)
      X <- mvrnorm(N, rep(0, p), Sigma_Design)
    }
    if (Sigma == 'Cons'){
      Sigma_Design <- cons_cov(p, r)
      X <- mvrnorm(N, rep(0, p), Sigma_Design)
    }
    Design <- X
  }
  support <- sample(2:p, s)
  mean_y <- magn * (0.5 * Design[,1]^2 + 
                      sin(3.1415 * Design[,1] / 2)) * (1 + rowSums(Design[,support]))
  
  if (Y_dist == 'Gaussian'){
    Y <- rnorm(N, mean_y, 1)
  }
  if (Y_dist == 'Binary'){
    Y <- ifelse(rnorm(N, mean_y, 1) > 0, 1, 0)
  }
  
  return(list(Y = Y, X = X, Sigma = Sigma_Design))
  
}
