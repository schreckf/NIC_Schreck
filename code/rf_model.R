############### Random forest model ###############################

set.seed(2610)

# Define task
rf_task    <- makeClassifTask(data = train[, c(vars$rf)], 
                           target = "customer", positive = "good")
  
# Define learner: decision tree
rf_learner <- makeLearner("classif.randomForest", 
                          predict.type = "prob",
                          par.vals = list("replace" = TRUE, 
                                          "importance" = FALSE, 
                                          "ntree" = 800)) 
  
# Tuning the hyperparameters of the random forest
rf_parms   <- makeParamSet(
  # Number of features selected at each node
  makeIntegerParam("mtry", lower = 2, upper = 12), 
  # Bootstrap sample size
  makeDiscreteParam("sampsize", values = c(30, 50, 70, 100, 130)),
  # Node size
  makeIntegerParam("nodesize", lower = 2, upper = 12) 
) 
rf_tunecontrol <- makeTuneControlGrid(resolution = 7, tune.threshold = FALSE) 
  
# Cross validation
rf_rdesc       <- makeResampleDesc(method = "CV", iters = 3, stratify = TRUE)
  
# Tuning with parallel computing
no_cores       <- detectCores() - 1 
  
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
rf_tuning_results <- generateHyperParsEffectData(rf_tuning, partial.dep = TRUE)
rf_tuning_results$data
  
# Detailed investigation
tapply(rf_tuning_results$data$auc.test.mean, 
       INDEX = c(rf_tuning_results$data$mtry), mean)
tapply(rf_tuning_results$data$auc.test.mean, 
       INDEX = c(rf_tuning_results$data$sampsize), mean)
tapply(rf_tuning_results$data$auc.test.mean, 
       INDEX = c(rf_tuning_results$data$nodesize), mean)
  
# Choose the optimal hyperparameters and update the learner
rf_tuned     <- setHyperPars(rf_learner, par.vals = rf_tuning$x)
rf_tuned
  
# Now we train the model on the full training data 
model_lib$rf <- mlr::train(rf_tuned, task = rf_task)
  
# Prediction on current test data
yhat$rf      <- predict(model_lib$rf, newdata = test)

