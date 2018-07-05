###### Neural net #################################################

set.seed(2610)
pred <- list()

# Define dataset
for (i in 1:5) {
  test_st       <- train_sets[[i]]
  train_st      <- train[-as.numeric(rownames(train_sets[[i]])), ]
    
  # Define task
  nnet_task     <- makeClassifTask(data = train_st[, c(vars$nnet)], 
                                   target = "customer", 
                                   positive = "good")
  
  # Define learner: neural net
  nnet_learner  <- makeLearner("classif.nnet", 
                               predict.type = "prob",
                               par.vals = list("trace" = FALSE, 
                                               "maxit" = 400, 
                                               "MaxNWts" = 3500)) 
  
  # Tuning the hyperparameters of the random forest
  nnet_parms    <- makeParamSet(
    makeDiscreteParam("decay", values = c(0, 0.1, 0.2, 0.3,
                                          0.4, 0.5, 0.6)), 
    makeDiscreteParam("size", values = c(2, 4, 5, 6, 7, 8,
                                         9, 10, 11, 12))
    )
  
  nnet_tunecontrol <- makeTuneControlGrid(resolution = 8, 
                                          tune.threshold = FALSE) 
  
  # Sampling strategy: cross validation
  nnet_rdesc       <- makeResampleDesc(method = "CV", 
                                       iters = 3, 
                                       stratify = TRUE)
  
  # Tuning with parallel computing
  no_cores         <- detectCores() - 1 # Detect number of cores
  
  parallelStartSocket(no_cores, level = "mlr.tuneParams")
  system.time(
    nnet_tuning    <- tuneParams(nnet_learner, 
                                 task = nnet_task, 
                                 resampling = nnet_rdesc, 
                                 par.set = nnet_parms, 
                                 control = nnet_tunecontrol, 
                                 measures = mlr::auc)
  )
  parallelStop()
  
  # Results for the different choices of hyperparameters
  nnet_tuning_results <- generateHyperParsEffectData(nnet_tuning, 
                                                     partial.dep = TRUE)
  nnet_tuning_results$data
  
  # Detailed investigation
  tapply(nnet_tuning_results$data$auc.test.mean, 
         INDEX = c(nnet_tuning_results$data$decay), mean)
  tapply(nnet_tuning_results$data$auc.test.mean, 
         INDEX = c(nnet_tuning_results$data$size), mean)
  
  # Choose the optimal hyperparameters and update the learner
  nnet_tuned     <- setHyperPars(nnet_learner, par.vals = nnet_tuning$x)
  nnet_tuned
  
  # Now we train the model on the full corresponding training data 
  model_lib$nnet <- mlr::train(nnet_tuned, task = nnet_task)
  
  # Prediction on current test_st data
  pred[[i]]      <- predict(model_lib$nnet, newdata = test_st)
}

# Combine subset predictions to obtain prediction on train dataset
yhat$nnet_train  <- rbind(pred[[1]]$data, pred[[2]]$data, pred[[3]]$data,
                    pred[[4]]$data, pred[[5]]$data)

# Prediction on test data 
yhat$nnet_test  <- predict(model_lib$nnet, newdata = test)

