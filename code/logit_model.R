###### Logit model ################################################

set.seed(2610)
pred <- list()

# Define dataset
for (i in 1:5) {
  test_st       <- train_sets[[i]]
  train_st      <- train[-as.numeric(rownames(train_sets[[i]])), ]
    
  # Define task
  logit_task    <- makeClassifTask(data = train_st[, c(vars$logit)], 
                                   target = "customer", 
                                   positive = "good")
  
  # Define learner: logistic regression
  logit_learner <- makeLearner("classif.logreg", 
                               predict.type = "prob") 
  
  # No tuning of hyperparameters necessary for logit model
  
  # Train the model on the full corresponding training data 
  model_lib$logit <- mlr::train(logit_learner, task = logit_task)
  
  # Prediction on current test_st data
  pred[[i]]       <- predict(model_lib$logit, newdata = test_st)
}  

# Combine subset predictions to obtain full prediction on train data
yhat$logit_train  <- rbind(pred[[1]]$data, pred[[2]]$data, pred[[3]]$data,
                           pred[[4]]$data, pred[[5]]$data)

# Prediction on test data 
yhat$logit_test   <- predict(model_lib$logit, newdata = test)

