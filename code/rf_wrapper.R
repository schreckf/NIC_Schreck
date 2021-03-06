###### Feature selection for the Random Forest model. A wrapper approach #####

# A model building approach with sequential forward selection is established in order
# to find the best subset of features. Each model is built with crossvalidation on 
# the AUC measure.

set.seed(2610) 

rf_task_wrapper     <- makeClassifTask(data = train,
                                       target = "customer",
                                       positive = "good")

rf_learner_wrapper  <- makeLearner("classif.randomForest", 
                                   predict.type = "prob",
                                   "ntree" = 300)

rf_ctrl_wrapper     <- makeFeatSelControlSequential(method = "sfs", 
                                                    alpha = 0.00001)

rf_rdesc_wrapper    <- makeResampleDesc("CV", iters = 3)

rf_sfeats           <- selectFeatures(learner = rf_learner_wrapper,
                                      task = rf_task_wrapper,
                                      resampling = rf_rdesc_wrapper,
                                      control = rf_ctrl_wrapper,
                                      show.info = TRUE,
                                      measures = mlr::auc)

# Performance score for each combination
analyzeFeatSelResult(rf_sfeats)

# Next, I define the dataset to use for the gradient boosting model.
# This dataset is then used in the model building file "rf_model.R"
vars$rf <- c("customer", rf_sfeats$x)

