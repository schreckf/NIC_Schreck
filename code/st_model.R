###### Stacked Generalization models ##############################

# Generate new train and test dataset with base learners' predictions as features
data_stacking_train <- as.data.frame(matrix(data = c(train$customer,
                                                     yhat$dt_train$prob.good,
                                                     yhat$logit_train$prob.good,
                                                     yhat$nnet_train$prob.good,
                                                     yhat$gb2_train$prob.good,
                                                     yhat$rf2_train$prob.good),
                                            nrow = NROW(train)), row.names = rownames(train))

data_stacking_test  <- as.data.frame(matrix(data = c(test$customer,
                                                     yhat$dt_test$data$prob.good,
                                                     yhat$logit_test$data$prob.good,
                                                     yhat$nnet_test$data$prob.good,
                                                     yhat$gb2_test$data$prob.good,
                                                     yhat$rf2_test$data$prob.good), 
                                            nrow = NROW(test)), row.names = rownames(test))

colnames(data_stacking_train) <- c("truth", "dt", 
                                   "logit", "nnet", 
                                   "gb", "rf")
colnames(data_stacking_test)  <- c("truth", "dt", 
                                   "logit", "nnet", 
                                   "gb", "rf")
data_stacking_train$truth     <- as.factor(data_stacking_train$truth)
data_stacking_test$truth      <- as.factor(data_stacking_test$truth)


# Investigate correlation of predictions and plot correlation matrix
cormat                        <- cor(data_stacking_train[2:6,2:6])

corrplot(cormat, type = "lower", method = "circle", order = "hclust", 
         col = rev(gray.colors(n = 90, start = 0.5, end = 1, gamma = 12)), 
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", cl.lim=c(0, 1))

###### Stacking Model 1: Averaging, all base learners ##############
yhat$st1           <- as.data.frame(matrix(data = c(data_stacking_test$truth, 
                                                    rowMeans(data_stacking_test[,-1])), 
                                           ncol = 2))
colnames(yhat$st1) <- c("truth", "prob.good")


###### Stacking Model 2: Averaging, least correlated learners ######
yhat$st2           <- as.data.frame(matrix(data = c(data_stacking_test$truth, 
                                                    rowMeans(data_stacking_test[,-c(1, 3, 5)])), 
                                           ncol = 2))
colnames(yhat$st2) <- c("truth", "prob.good")


###### Stacking Model 3: Random Forest, all base learners ##########
set.seed(2610)
# Define task
rf_task        <- makeClassifTask(data = data_stacking_train, 
                                  target = "truth", positive = "1")

# Define learner: decision tree
rf_learner     <- makeLearner("classif.randomForest", 
                          predict.type = "prob",
                          par.vals = list("replace" = TRUE, 
                                          "importance" = FALSE, 
                                          "ntree" = 800)) 

# Tuning the hyperparameters of the random forest
rf_parms       <- makeParamSet(
  # Number of features selected at each node
  makeIntegerParam("mtry", lower = 2, upper = 6),  
  # Bootstrap sample size 
  makeDiscreteParam("sampsize", values = c(30, 50, 70, 100, 130)), 
  # Size of nodes
  makeIntegerParam("nodesize", lower = 3, upper = 5) 
) 
rf_tunecontrol <- makeTuneControlGrid(resolution = 5, 
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

# Choose the optimal hyperparameters and update the learner
rf_tuned          <- setHyperPars(rf_learner, par.vals = rf_tuning$x)
rf_tuned

# Now we train the model on the full training data (no crossvalidation)
model_lib$st3     <- mlr::train(rf_tuned, task = rf_task)

# Prediction on current test data
yhat$st3          <- predict(model_lib$st3, newdata = data_stacking_test)


###### Stacking Model 4: Random Forest, least correlated learners #####
set.seed(2610)
# Define task
rf_task        <- makeClassifTask(data = data_stacking_train[, -c(3,5)], 
                                  target = "truth", positive = "1")

# Define learner: decision tree
rf_learner     <- makeLearner("classif.randomForest", 
                          predict.type = "prob",
                          par.vals = list("replace" = TRUE, 
                                          "importance" = FALSE, 
                                          "ntree" = 800)) 

# Tuning the hyperparameters of the random forest
rf_parms       <- makeParamSet(
  # Number of features selected at each node.
  makeIntegerParam("mtry", lower = 2, upper = 6),  
  # Bootstrap sample size
  makeDiscreteParam("sampsize", values = c(30, 50, 70, 100, 130)),  
  # Size of nodes
  makeIntegerParam("nodesize", lower = 3, upper = 5) 
) 
rf_tunecontrol <- makeTuneControlGrid(resolution = 5, 
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

# Choose the optimal hyperparameters and update the learner
rf_tuned          <- setHyperPars(rf_learner, par.vals = rf_tuning$x)
rf_tuned

# Now we train the model on the full training data (no crossvalidation)
model_lib$st4     <- mlr::train(rf_tuned, task = rf_task)

# Prediction on current test data
yhat$st4          <- predict(model_lib$st4, newdata = data_stacking_test[, -c(1,3,5)])



