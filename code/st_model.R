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
colnames(cormat) <- c("Decision Tree", "Logistic Regression", 
                      "Neural Network", "Gradient Boosting", 
                      "Random Forest")
rownames(cormat) <- c("Decision Tree", "Logistic Regression", 
                      "Neural Network", "Gradient Boosting", 
                      "Random Forest")
X11(width=6, height=6)
corrplot(cormat, type = "lower", method = "circle", order = "hclust", 
         col = rev(gray.colors(n = 90, start = 0.5, end = 1, gamma = 12)), 
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black")

###### Stacking Model 1: Averaging, all base learners ##############
yhat$st1           <- as.data.frame(matrix(data = c(data_stacking_test$truth, 
                                                    rowMeans(data_stacking_test[,-1])), 
                                           ncol = 2))
colnames(yhat$st1) <- c("truth", "prob.good")


###### Stacking Model 2: Averaging, best learners ######
yhat$st2           <- as.data.frame(matrix(data = c(data_stacking_test$truth, 
                                                    rowMeans(data_stacking_test[,-c(1, 2, 3)])), 
                                           ncol = 2))
colnames(yhat$st2) <- c("truth", "prob.good")


###### Stacking Model 3: Gradient Boosting, all base learners ##########
set.seed(2610)

control <- trainControl(method="repeatedcv", number=10, 
                        repeats=3, savePredictions=TRUE, 
                        classProbs=TRUE)
algorithms <- c('glm', 'rpart', 'rf', 'nnet', 'xgbLinear')
base <- caretList(customer~., data=train, trControl=control, 
                  methodList=algorithms)

# Stacker
stackControl <- trainControl(method="repeatedcv", number=10, 
                             repeats=3, savePredictions=TRUE, 
                             classProbs=TRUE,
                             summaryFunction = twoClassSummary)
model_lib$st3 <- caretStack(base, method="gbm", metric="ROC", 
                            trControl=stackControl)

# Prediction on test dataset
yhat$st3          <- matrix(c(rownames(test),
                              test$customer,
                              1 - predict(model_lib$st3,
                                          newdata=test,
                                          type="prob")),
                            ncol = 3)
yhat$st3 <- apply(yhat$st3, 2, function(x) as.numeric(x))
colnames(yhat$st3) <- c("id", "truth", "prob.good")


###### Stacking Model 4: Logistic Regression, all base learners #########
set.seed(2610)

# Stacker
model_lib$st4 <- caretStack(base, method="glmnet",
                            metric="ROC", trControl=stackControl)

# Prediction on test dataset
yhat$st4          <- matrix(c(rownames(test),
                              test$customer,
                              1 - predict(model_lib$st4,
                                          newdata=test,
                                          type="prob")),
                            ncol = 3)
yhat$st4 <- apply(yhat$st4, 2, function(x) as.numeric(x))
colnames(yhat$st4) <- c("id", "truth", "prob.good")
