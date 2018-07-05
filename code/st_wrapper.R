###### Variable selection for the ensemble stacking model. Wrapper approach with mlr package #####

# We establish a model building approach with sequential forward search in order
# to find the best subset of features.
# Hereby we choose the tuned parameters from the subsequently established learners.
# We calculate different stackers with 3-fold-crossvalidation and use the 
# auc values of the models to choose best subset of data.
# Since we use one-hot-encoded categorial features, the corresponding dummies
# are evaluated separately

set.seed(123) 

# One-hot-encoding
known_wrapper <- mlr::createDummyFeatures(known_vs, target = "return") 

# Since XGBOOST has problems with integer variables we redefine them as factors
integer_vars                  <- sapply(known_wrapper, is.integer)
known_wrapper[, integer_vars] <- lapply(known_wrapper[, integer_vars], as.numeric)
known_wrapper$return          <- as.factor(known_wrapper$return)

# Define task
st_task_wrapper <- makeClassifTask(data = known_wrapper, target = "return", positive = "1")

# Define base learners: neural net, gradient boosting model and random forest, 
st_learner1_wrapper <- makeLearner("classif.nnet", 
                                   predict.type = "prob",
                                   par.vals = list("trace" = FALSE, 
                                                   "maxit" = 400, 
                                                   "MaxNWts" = 3500,
                                                   "decay" = 0.11,
                                                   "size" = 12))

st_learner2_wrapper <- makeLearner("classif.xgboost", 
                                   predict.type = "prob",
                                   par.vals = list("booster" = "gbtree",
                                                   "silent" = 0,
                                                   "eta" = 0.45, 
                                                   "max_depth" = 6,
                                                   "min_child_weight" = 4,
                                                   "nrounds" = 10,
                                                   "lambda" = 0.15, 
                                                   "gamma" = 0.5,
                                                   "subsample" = 1))

st_base_wrapper <- list(st_learner1_wrapper, st_learner2_wrapper)

# Define stacked learner
st_learner_wrapper <- makeStackedLearner(base.learners = st_base_wrapper,
                                        predict.type = "prob",
                                        method = "hill.climb")

# Set feature selection method and set the minimum required difference to 
# 0.0001 Euros in expected loss per observation
st_ctrl_wrapper <- makeFeatSelControlSequential(method = "sfs", alpha = 0.0001) 

# Sampling Strategy: cross-validation
st_rdesc_wrapper <- makeResampleDesc("CV", iters = 2)

# Extract best subset of features
st_sfeats <- selectFeatures(learner = st_learner_wrapper, 
                            task = st_task_wrapper,
                            resampling = st_rdesc_wrapper,
                            control = st_ctrl_wrapper,
                            show.info = FALSE, 
                            measures = loss_measure)

# Performance score for each combination
analyzeFeatSelResult(st_sfeats)

# We now define the dataset to use for the stacking model.
# This dataset is then used in the model building file "st_model.R"
vars$st <- known_wrapper[, c("return", st_sfeats$x)]

