################ Decision tree model ##############################

set.seed(2610)
pred <- list()

# Define dataset
for (i in 1:5) {
  test_st    <- train_sets[[i]]
  train_st   <- train[-as.numeric(rownames(train_sets[[i]])), ]

  # Define task
  dt_task    <- makeClassifTask(data = train_st[, c(vars$dt)], 
                                target = "customer", 
                                positive = "good")
  
  # Define learner: decision tree
  dt_learner <- makeLearner("classif.rpart", 
                            predict.type = "prob") 
  
  # Tuning the hyperparameter of the tree
  dt_parms   <- makeParamSet(
    # Complexity parameter
    makeNumericParam("cp", lower = 0.00005, upper = 0.001), 
    # Minimum number of observation in a node for a split
    makeDiscreteParam("minsplit", values = c(5, 7, 10, 12, 15, 17, 
                                             20, 23, 25, 27, 30)),  
    # Minimum number of observation to keep in terminal nodes
    makeDiscreteParam("minbucket", values = c(5, 8, 10, 12, 
                                              15, 18, 21, 25)) 
  )  
  
  # Grid density
  dt_tunecontrol <- makeTuneControlGrid(resolution = 10, 
                                        tune.threshold = FALSE)
  
  # Sampling strategy: cross validation
  dt_rdesc       <- makeResampleDesc(method = "CV", 
                                     iters = 3, 
                                     stratify = TRUE)
  
  # Tuning using parallelization
  no_cores       <- detectCores() - 1 # Detect number of cores
  
  parallelStartSocket(no_cores, level = "mlr.tuneParams")
  system.time(
    dt_tuning    <- tuneParams(dt_learner, 
                               task = dt_task, 
                               resampling = dt_rdesc,
                               par.set = dt_parms,
                               control = dt_tunecontrol,
                               measures = mlr::auc)
  )
  parallelStop()
  
  # Results for the different choices of hyperparameters
  dt_tuning_results <- generateHyperParsEffectData(dt_tuning, 
                                                   partial.dep = TRUE)
  dt_tuning_results$data
  
  # Detailed investigation
  tapply(dt_tuning_results$data$auc.test.mean, 
         INDEX = c(dt_tuning_results$data$cp), mean)
  tapply(dt_tuning_results$data$auc.test.mean, 
         INDEX = c(dt_tuning_results$data$minsplit), mean)
  tapply(dt_tuning_results$data$auc.test.mean, 
         INDEX = c(dt_tuning_results$data$minbucket), mean)
  
  # Choose the optimal hyperparameters and update the learner
  dt_tuned     <- setHyperPars(dt_learner, par.vals = dt_tuning$x)
  dt_tuned
  
  # Now the model is trained on the corresponding training data 
  model_lib$dt <- mlr::train(dt_tuned, task = dt_task)
  
  # Prediction on current test_st data
  pred[[i]]    <- predict(model_lib$dt, newdata = test_st)
}

# Combine subset predictions to obtain prediction on train dataset
yhat$dt_train  <- rbind(pred[[1]]$data, pred[[2]]$data, pred[[3]]$data, 
                        pred[[4]]$data, pred[[5]]$data)

# Prediction on test data 
yhat$dt_test   <- predict(model_lib$dt, newdata = test)

