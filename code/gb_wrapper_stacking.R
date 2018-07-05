###### Variable selection for the Gradient Boosting model. A wrapper approach #####

# A model building approach with sequential forward selection is 
# established in order to find the best subset of features. 
# Each model is built with crossvalidation on the AUC measure.

# One-hot-encoding of categorical features
train_wrapper2        <- mlr::createDummyFeatures(train, target = "customer") 

gb_task_wrapper       <- makeClassifTask(data = train_wrapper2, 
                                         target = "customer", 
                                         positive = "good")

gb_learner_wrapper    <- makeLearner("classif.xgboost",
                                     predict.type = "prob") 

gb_ctrl_wrapper       <- makeFeatSelControlSequential(method = "sfs",
                                                      alpha = 0.00001) 

gb_rdesc_wrapper      <- makeResampleDesc("CV", iters = 3)

gb_sfeats             <- selectFeatures(learner = gb_learner_wrapper,
                                        task = gb_task_wrapper,
                                        resampling = gb_rdesc_wrapper,
                                        control = gb_ctrl_wrapper,
                                        show.info = TRUE, 
                                        measures = mlr::auc)

# Performance score for each combination of features
analyzeFeatSelResult(gb_sfeats) 

# Next, I store the optimal set of features to later use it 
# in the model building part file  "gb_model_stacking.R"
vars$gb2 <- c("customer", gb_sfeats$x)

