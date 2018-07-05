###### Stacking: Gradient Boosting model (with mlr package) #######

set.seed(2610)
pred <- list()

# Define dataset
for (i in 1:5) {
  test_st     <- train_sets[[i]]
  train_st    <- train[-as.numeric(rownames(train_sets[[i]])), ]
  
  test_st     <- mlr::createDummyFeatures(test_st, target = "customer")
  train_st    <- mlr::createDummyFeatures(train_st, target = "customer")
  # Define task
  gb_task     <- makeClassifTask(data = train_st[, c(vars$gb2)], 
                                 target = "customer", 
                                 positive = "good")
  
  # Define learner: gradient boosting model consisting of trees
  gb_learner  <- makeLearner("classif.xgboost", 
                             predict.type = "prob",
                             par.vals = list("booster" = "gbtree", 
                                             "silent" = 0))
  
  # Tuning the hyperparameter of the model
  gb_parms    <- makeParamSet(
    # Learning rate
    makeDiscreteParam("eta", values = c(0.35, 0.45, 0.5, 0.55, 0.6)), 
    # Maximum depth of a tree
    makeIntegerParam("max_depth", lower = 3, upper = 10), 
    # Minimum number of obs. to have per node
    makeIntegerParam("min_child_weight", lower = 2, upper = 4),
    # Number of iterations through data
    makeIntegerParam("nrounds", lower = 8, upper = 16), 
    # L2 regularization on weights
    makeDiscreteParam("lambda", values = c(0.05, 0.1, 0.15, 0.2, 0.3)),  
    # Minimum loss reduction
    makeDiscreteParam("gamma", values = c(0.3, 0.4, 0.5, 0.6)), 
    # Subsample size
    makeDiscreteParam("subsample", values = c(0.9, 0.95, 1))
  )  
  
  # Define how dense the parameters areselected from the defined ranges
  gb_tunecontrol <- makeTuneControlGrid(resolution = 3, 
                                        tune.threshold = FALSE)
  
  # Sampling strategy: cross validation
  gb_rdesc       <- makeResampleDesc(method = "CV", 
                                     iters = 5, 
                                     stratify = TRUE)
  
  # Tuning
  no_cores       <- detectCores() - 1 # Detect number of cores
  
  parallelStartSocket(no_cores, level = "mlr.tuneParams")
  system.time(
    gb_tuning    <- tuneParams(gb_learner, 
                               task = gb_task,
                               resampling = gb_rdesc,
                               par.set = gb_parms, 
                               control = gb_tunecontrol,
                               measures = mlr::auc)
  )
  parallelStop()
  
  # Results for the different choices of hyperparameters
  gb_tuning_results <- generateHyperParsEffectData(gb_tuning, 
                                                   partial.dep = TRUE)
  gb_tuning_results$data
  
  # Detailed investigation
  tapply(gb_tuning_results$data$auc.test.mean, 
         INDEX = c(gb_tuning_results$data$eta), mean)
  tapply(gb_tuning_results$data$auc.test.mean, 
         INDEX = c(gb_tuning_results$data$min_child_weight), mean)
  tapply(gb_tuning_results$data$auc.test.mean, 
         INDEX = c(gb_tuning_results$data$nrounds), mean)
  tapply(gb_tuning_results$data$auc.test.mean, 
         INDEX = c(gb_tuning_results$data$lambda), mean)
  tapply(gb_tuning_results$data$auc.test.mean, 
         INDEX = c(gb_tuning_results$data$gamma), mean)
  tapply(gb_tuning_results$data$auc.test.mean, 
         INDEX = c(gb_tuning_results$data$subsample), mean)
  tapply(gb_tuning_results$data$auc.test.mean, 
         INDEX = c(gb_tuning_results$data$max_depth), mean)
  
  # Choose the optimal hyperparameters and update the learner
  gb_tuned      <- setHyperPars(gb_learner, par.vals = gb_tuning$x)
  gb_tuned
  
  # Now we train the model on the full training data 
  model_lib$gb2 <- mlr::train(gb_tuned, task = gb_task)
  
  # Prediction on current test_st data
  pred[[i]]     <- predict(model_lib$gb2, newdata = test_st[, c(vars$gb2)])
}

# Combine subset predictions to obtain full prediction on train data
yhat$gb2_train <- rbind(pred[[1]]$data, pred[[2]]$data, pred[[3]]$data, 
                        pred[[4]]$data, pred[[5]]$data)

# Performance measured on one-hot encoded test data 
test_onehot  <- mlr::createDummyFeatures(test, target = "customer") 
yhat$gb2_test  <- predict(model_lib$gb2, newdata = test_onehot[, c(vars$gb2)])

