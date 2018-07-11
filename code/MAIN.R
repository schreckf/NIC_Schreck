###################################################################
# Numerical Introductory Course 2018
# Topic: Stacking and Ensemble Modelling
# Supervisor: Prof. Dr. Brenda L?pez Cabrera
# Student: Frederik Schreck
###################################################################


#------------------------------------------------------------------
# Prepare environment
#------------------------------------------------------------------

# Install packages and set working directory to path of main.R file
library("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library("plyr")
library("mlr")
library("parallelMap")
library("parallel")
library("ggplot2")
library("corrplot")
library("glmnet")
library("xtable")
library("caret")
library("caretEnsemble")
library("gbm")

rm(list = ls(all = TRUE))
graphics.off()


#------------------------------------------------------------------
# Loading dataset: German Credit Data
#------------------------------------------------------------------

creditdata           <- read.delim("dataset/german_credit_data.txt", 
                                   header = FALSE, sep = " ")

# Renaming of feature
colnames(creditdata) <- c("account_status", "duration", 
                           "credit_history", "purpose", "amount", 
                           "savings", "employment_duration", 
                           "installment_rate", "status_sex", 
                           "other_debtors", "residence_duration", 
                           "property", "age", "installment_plans", 
                           "housing", "number_credits", "job", 
                           "liable_people", "telephone", 
                           "foreign_worker", "customer")


#------------------------------------------------------------------
# Feature Engineering
#------------------------------------------------------------------

# Formatting and labelling
creditdata$duration           <- as.numeric(creditdata$duration)
creditdata$amount             <- as.numeric(creditdata$amount)
creditdata$installment_rate   <- as.numeric(creditdata$installment_rate)
creditdata$residence_duration <- as.numeric(creditdata$residence_duration)
creditdata$age                <- as.numeric(creditdata$age)
creditdata$number_credits     <- as.numeric(creditdata$number_credits)
creditdata$duration           <- as.numeric(creditdata$duration)
creditdata$liable_people      <- as.numeric(creditdata$liable_people)
creditdata$customer           <- factor(creditdata$customer, 
                                        levels = c(1, 2), 
                                        labels = c("good", "bad"))

# No problem of missing values
apply(creditdata, 2, function(x) sum(is.na(x)))

# Create summary tables for numeric and for categorical features
numeric_vars <- lapply(Filter(is.numeric, creditdata), 
                       function(x) rbind(mean = mean(x),
                                         sd = sd(x),
                                         median = median(x),
                                         minimum = min(x),
                                         maximum = max(x)))
print(xtable(t(data.frame(numeric_vars))), 
      file="tables/summary_numeric.txt")


cat_vars <- ldply(Filter(is.factor, creditdata), 
                  function(x) t(rbind(names(table(x)),
                                      table(x),
                                      paste0(prop.table(table(x))*100,"%"))))
colnames(cat_vars) <- c("feature", "category", "Count", "Fraction")
print(xtable(data.frame(cat_vars)), 
      file="tables/summary_categorical.txt")

# Standardize numeric features for models performance
creditdata$duration           <- scale(creditdata$duration)
creditdata$amount             <- scale(creditdata$amount)
creditdata$installment_rate   <- scale(creditdata$installment_rate)
creditdata$residence_duration <- scale(creditdata$residence_duration)
creditdata$age                <- scale(creditdata$age)
creditdata$number_credits     <- scale(creditdata$number_credits)
creditdata$duration           <- scale(creditdata$duration)
creditdata$liable_people      <- scale(creditdata$liable_people)


#------------------------------------------------------------------
# Data partitioning into training and testing data
#------------------------------------------------------------------

set.seed(2601)
idx   <- sample(x = NROW(creditdata),
                size = floor(0.75*NROW(creditdata)), 
                replace = FALSE)
train <- creditdata[idx,]
test  <- creditdata[-idx,]


#------------------------------------------------------------------
# Feature Selection
#------------------------------------------------------------------

'For each model, a wrapper approach is run on the training dataset 
in order to find the best subset of features. The optimal set of
features is the then used in the subsequent model building part. 

The wrappers as well as the model tuning are evaluated on the AUC
measure. Tuning of the hyperparameters in the wrapper is done by 
a grid-approach with 3-fold cross-validation process. 

For reproduction of the results, the output of the wrapper 
files can be accessed directly.'

# Generate empty lists to store the models and their results in
vars       <- list() # Selected set of features used for each model
model_lib  <- list() # Model library
yhat       <- list() # Predictions on test dataset


##### Gradient Boosting wrapper ###################################
source("gb_wrapper.R")

# For replication purposes: gb_wrapper results
train_wrapper <- mlr::createDummyFeatures(train, target = "customer")
vars$gb       <- c("customer", "account_status.A11", 
                   "account_status.A14", "credit_history.A30",
                   "credit_history.A34", "purpose.A49", "savings.A63",
                   "savings.A65", "employment_duration.A75",
                   "other_debtors.A103", "property.A123",
                   "installment_plans.A143", "job.A174",
                   "telephone.A192")       


##### Random forest wrapper #######################################
source("rf_wrapper.R")

# For replication purposes: rf_wrapper results
vars$rf      <- c("customer", "account_status", "duration", 
                  "credit_history", "purpose", "savings",
                  "employment_duration", "other_debtors",
                  "residence_duration", "age", "installment_plans",
                  "housing", "job", "liable_people", "foreign_worker")


##### Stacked Generalization model ################################
'For the Stacked Generalization model, firstly the training data is
split into five disjoint sets. Secondly, the Random Forest model and 
the Gradient Boosting model are rebuilt on that subset. For each 
model, a new wrapper approach is applied to find the optimal subset
of features.

Additionally, a Decision Tree, a Logistic Regression and a 
Neural Network are built. Similar to the models before, a 
model-specific wrapper approach is applied before the model
building process.

For reproduction of the results, the output of the wrapper 
functions can again be accessed directly.
'

##### Stacking: Decision tree wrapper #############################
source("dt_wrapper.R")

# For replication purposes: dt_wrapper results
vars$dt       <- c("customer", "account_status", "credit_history",
                   "savings", "number_credits")

##### Stacking: Logit wrapper #####################################
source("logit_wrapper.R")

# For replication purposes: logit_wrapper results
vars$logit    <- c("customer", "duration", "credit_history",
                   "purpose", "amount", "installment_rate",
                   "status_sex", "residence_duration", "age",
                   "installment_plans", "number_credits",
                   "liable_people", "telephone", "foreign_worker")  

##### Stacking: Neural net wrapper ################################
source("nnet_wrapper.R")

# For replication purposes: nnet_wrapper results
vars$nnet     <- c("customer", "account_status", "credit_history")

##### Stacking: Gradient Boosting wrapper #########################
source("gb_wrapper_stacking.R") 

# For replication purposes: gb_wrapper results
vars$gb2      <- c("customer", "account_status.A11", 
                   "account_status.A14", "credit_history.A34",
                   "purpose.A41", "savings.A65", 
                   "employment_duration.A74", "status_sex.A94",
                   "other_debtors.A103", "installment_plans.A143")      

##### Stacking: Random forest wrapper #############################
source("rf_wrapper_stacking.R")

# For replication purposes: rf_wrapper results
vars$rf2      <- c("customer", "account_status", "duration",
                   "credit_history", "purpose", "savings",
                   "employment_duration", "other_debtors",
                   "residence_duration", "age",
                   "installment_plans", "housing", "job",
                   "liable_people", "foreign_worker")


#------------------------------------------------------------------
# Model Building
#------------------------------------------------------------------

'In the following, the models are built, parameters are tuned and 
predictions on the test dataset are made. Each model uses the 
optimal subset of features from  the corresponding wrapper approach.'

###### Random forest tuning #######################################
source("rf_model.R")


###### Gradient Boosting tuning ###################################
source("gb_model.R")


###### Stacking: preparation ######################################

# Partitioning the dataset for the Stacking into five equally 
# sized disjoint sets
set.seed(2610)
idx2       <- sample(rep(1:5,each = nrow(train)/5)) 
train_sets <- lapply(split(1:nrow(train), idx2), 
                     function(i) creditdata[i,])

###### Stacking: Decision tree tuning #############################
source("dt_model.R")


###### Stacking: Logit model ######################################
# No tuning necessary 
source("logit_model.R")


###### Stacking: Neural net tuning ################################
source("nnet_model.R")


###### Stacking: Gradient Boosting tuning #########################
source("gb_model_stacking.R")


###### Stacking: Random forest tuning #############################
source("rf_model_stacking.R")


###### Stacked Generalization Models ##############################
source("st_model.R")


#------------------------------------------------------------------
# Model evaluation
#------------------------------------------------------------------

'After having built all the models, they can now be compared and 
evaluated with regard to a variety of evaluation metrics.'

# Create empty lists to store the measure values in
auc        <- list() # Area under curve performance measure
acc        <- list() # Accuracy performance measure
kappa      <- list() # Kappa performance measure
logloss    <- list() # Logarithmic Loss measure
brier      <- list() # Brier score measure 

# AUC
auc$rf     <- mlr::performance(yhat$rf, 
                               measures = mlr::auc); auc$rf
auc$gb     <- mlr::performance(yhat$gb, 
                               measures = mlr::auc); auc$gb
auc$dt     <- mlr::performance(yhat$dt_test, 
                               measures = mlr::auc); auc$dt
auc$logit  <- mlr::performance(yhat$logit_test, 
                               measures = mlr::auc); auc$logit
auc$nnet   <- mlr::performance(yhat$nnet_test, 
                               measures = mlr::auc); auc$nnet
auc$gb2    <- mlr::performance(yhat$gb2_test, 
                               measures = mlr::auc); auc$gb2
auc$rf2    <- mlr::performance(yhat$rf2_test, 
                               measures = mlr::auc); auc$rf2
auc$st1    <- measureAUC(probabilities = yhat$st1$prob.good, 
                      truth = yhat$st1$truth, positive = "1", 
                      negative = "2"); auc$st1
auc$st2    <- measureAUC(probabilities = yhat$st2$prob.good, 
                      truth = yhat$st2$truth, positive = "1", 
                      negative = "2"); auc$st2
auc$st3    <- measureAUC(probabilities = yhat$st3[,3], 
                         truth = yhat$st3[,2], positive = "1", 
                         negative = "2"); auc$st3
auc$st4    <- measureAUC(probabilities = yhat$st4[,3], 
                         truth = yhat$st4[,2], positive = "1", 
                         negative = "2"); auc$st4

# Accuracy
acc$rf     <- mlr::performance(yhat$rf, 
                               measures = mlr::acc); acc$rf
acc$gb     <- mlr::performance(yhat$gb, 
                               measures = mlr::acc); acc$gb
acc$dt     <- mlr::performance(yhat$dt_test, 
                               measures = mlr::acc); acc$dt
acc$logit  <- mlr::performance(yhat$logit_test, 
                               measures = mlr::acc); acc$logit
acc$nnet   <- mlr::performance(yhat$nnet_test,
                               measures = mlr::acc); acc$nnet
acc$gb2    <- mlr::performance(yhat$gb2_test,
                               measures = mlr::acc); acc$gb2
acc$rf2    <- mlr::performance(yhat$rf2_test, 
                               measures = mlr::acc); acc$rf2
acc$st1    <- measureACC(response = round(yhat$st1$prob.good), 
                         truth = yhat$st1$truth); acc$st1
acc$st2    <- measureACC(response = round(yhat$st2$prob.good), 
                         truth = yhat$st2$truth); acc$st2
acc$st3    <- measureACC(response = round(yhat$st3[,3]), 
                         truth = yhat$st3[,2]); acc$st3
acc$st4    <- measureACC(response = round(yhat$st4[,3]), 
                         truth = yhat$st4[,2]); acc$st4

# Logarithmic Loss
logloss$rf    <- mlr::performance(yhat$rf, 
                                  measures = mlr::logloss)
logloss$rf
logloss$gb    <- mlr::performance(yhat$gb, 
                                  measures = mlr::logloss)
logloss$gb
logloss$dt    <- mlr::performance(yhat$dt_test, 
                                  measures = mlr::logloss)
logloss$dt
logloss$logit <- mlr::performance(yhat$logit_test,
                                  measures = mlr::logloss)
logloss$logit
logloss$nnet  <- mlr::performance(yhat$nnet_test, 
                                  measures = mlr::logloss)
logloss$nnet
logloss$gb2   <- mlr::performance(yhat$gb2_test, 
                                  measures = mlr::logloss)
logloss$gb2
logloss$rf2   <- mlr::performance(yhat$rf2_test, 
                                  measures = mlr::logloss)
logloss$rf2
log_loss=function(actual, predicted)
{
  result=-1/length(actual)*
    (sum((actual*log(predicted)+
            (1-actual)*log(1-predicted))))
  return(result)
}
logloss$st1   <- log_loss(yhat$st1$truth, yhat$st1$prob.good)
logloss$st1
logloss$st2   <- log_loss(yhat$st2$truth, yhat$st2$prob.good)
logloss$st2
logloss$st3   <- log_loss(yhat$st3[,2], yhat$st3[,3])
logloss$st3
logloss$st4   <- log_loss(yhat$st4[,2], yhat$st4[,3])
logloss$st4

# MSE/Brier score
brier$rf      <- mlr::performance(yhat$rf, 
                                  measures = mlr::brier); brier$rf
brier$gb      <- mlr::performance(yhat$gb, 
                                  measures = mlr::brier); brier$gb
brier$dt      <- mlr::performance(yhat$dt_test, 
                                  measures = mlr::brier); brier$dt
brier$logit   <- mlr::performance(yhat$logit_test,
                                  measures = mlr::brier); brier$logit
brier$nnet    <- mlr::performance(yhat$nnet_test, 
                                  measures = mlr::brier); brier$nnet
brier$gb2     <- mlr::performance(yhat$gb2_test, 
                                  measures = mlr::brier); brier$gb2
brier$rf2     <- mlr::performance(yhat$rf2_test, 
                                  measures = mlr::brier); brier$rf2
brier$st1     <- measureBrier(probabilities = yhat$st1$prob.good, 
                              truth = yhat$st1$truth, 
                              positive = 1, negative = 2); brier$st1
brier$st2     <- measureBrier(probabilities = yhat$st2$prob.good, 
                              truth = yhat$st2$truth, positive = 1, 
                              negative = 2); brier$st2
brier$st3     <- measureBrier(probabilities = yhat$st3[,3], 
                              truth = yhat$st3[,2], positive =1, 
                              negative = 2); brier$st3
brier$st4     <- measureBrier(probabilities = yhat$st4[,3], 
                              truth = yhat$st4[,2], positive = 1, 
                              negative = 2); brier$st4


# Make evaluation table
eval_table           <- cbind(auc, acc, logloss, brier)
rownames(eval_table) <- c("Random Forest", "Gradient Boosting", 
                          "Decision Tree (level 0)", 
                          "Logit Regression (level 0)",
                          "Neural Network (level 0)", 
                          "Random Forest (level 0)", 
                          "Gradient Boosting (level 0)",
                          "Stacking Model 1", "Stacking Model 2", 
                          "Stacking Model 3", "Stacking Model 4")
colnames(eval_table) <- c("AUC", "Accuracy", 
                          "Logarithmic Loss", "Brier Score")


print(xtable(eval_table), file="tables/evaltable.txt")


