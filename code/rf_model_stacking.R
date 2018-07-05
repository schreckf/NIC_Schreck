###### Stacking: Random forest model ##############################

set.seed(2610)
pred <- list()

# Define dataset
for (i in 1:5) {
  test_st        <- train_sets[[i]]
  train_st       <- train[-as.numeric(rownames(train_sets[[i]])), ]
  
  # Define task
  rf_task        <- makeClassifTask(data = train_st[, c(vars$rf2)], 
                                    target = "customer", 
                                    positive = "good")
  
  # Define learner: decision tree
  rf_learner     <- makeLearner("classif.randomForest",
                                predict.type = "prob",
                                par.vals = list("replace" = TRUE, 
                                                "importance" = FALSE, 
                                                "ntree" = 800)) 
  
  # Tuning the hyperparameters of the random forest
  rf_parms       <- makeParamSet(
    # Number of features selected at each node. 
    makeIntegerParam("mtry", lower = 2, upper = 12), 
    # Bootstrap sample size 
    makeDiscreteParam("sampsize", values = c(30, 50, 70, 100, 130)), 
    # Size of nodes
    makeIntegerParam("nodesize", lower = 2, upper = 12) 
  ) 
  # Grid density
  rf_tunecontrol <- makeTuneControlGrid(resolution = 7, 
                                        tune.threshold = FALSE) 
  
  # Sampling strategy: cross validation
  rf_rdesc       <- makeResampleDesc(method = "CV", 
                                     iters = 3, 
                                     stratify = TRUE)
  
  # Tuning with parallel computing
  no_cores       <- detectCores() - 1 # Detect number of cores
  
  parallelStartSocket(no_cores, level = "mlr.tuneParams")
  system.time(
    rf_tuning    <- tuneParams(rf_learner, task = rf_task, 
                               resampling = rf_rdesc, 
                               par.set = rf_parms, 
                               control = rf_tunecontrol, 
                               measures = mlr::auc)
  )
  parallelStop()
  
  # Results for the different choices of hyperparameters
  rf_tuning_results <- generateHyperParsEffectData(rf_tuning, 
                                                   partial.dep = TRUE)
  rf_tuning_results$data
  
  # Detailed investigation
  tapply(rf_tuning_results$data$auc.test.mean, 
         INDEX = c(rf_tuning_results$data$mtry), mean)
  tapply(rf_tuning_results$data$auc.test.mean, 
         INDEX = c(rf_tuning_results$data$sampsize), mean)
  tapply(rf_tuning_results$data$auc.test.mean, 
         INDEX = c(rf_tuning_results$data$nodesize), mean)
  
  # Choose the optimal hyperparameters and update the learner
  rf_tuned       <- setHyperPars(rf_learner, par.vals = rf_tuning$x)
  rf_tuned
  
  # Now we train the model on the full training data 
  model_lib$rf2  <- mlr::train(rf_tuned, task = rf_task)
  
  # Prediction on current test_st data
  pred[[i]]      <- predict(model_lib$rf2, newdata = test_st)
}

# Combine subset predictions to obtain prediction on full train data
yhat$rf2_train  <- rbind(pred[[1]]$data, pred[[2]]$data, pred[[3]]$data,
                         pred[[4]]$data, pred[[5]]$data)

# Performance measured on test data 
yhat$rf2_test   <- predict(model_lib$rf2, newdata = test)
