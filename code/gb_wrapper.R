###### Feature selection for the Gradient Boosting model. A wrapper approach #####

# A model building approach with sequential forward selection is 
# established in order to find the best subset of features. 
# Each model is built with crossvalidation on the AUC measure.

set.seed(2610) 

# One-hot-encoding of categorical features
train_wrapper     <- mlr::createDummyFeatures(train, 
                                              target = "customer") 

gb_task_wrapper   <- makeClassifTask(data = train_wrapper, 
                                     target = "customer", 
                                     positive = "good")

gb_learner_wrapper<- makeLearner("classif.xgboost",
                                 predict.type = "prob") 

gb_ctrl_wrapper   <- makeFeatSelControlSequential(method = "sfs", 
                                                  alpha = 0.00001) 

gb_rdesc_wrapper  <- makeResampleDesc("CV", iters = 3)

gb_sfeats         <- selectFeatures(learner = gb_learner_wrapper,
                                    task = gb_task_wrapper,
                                    resampling = gb_rdesc_wrapper,
                                    control = gb_ctrl_wrapper,
                                    show.info = TRUE, 
                                    measures = mlr::auc)

# Performance score for each combination of features
analyzeFeatSelResult(gb_sfeats) 

# Next, I define the dataset to use for the gradient 
# boosting model. This dataset is then used in the model 
# building file "gb_model.R"
vars$gb          <- c("customer", gb_sfeats$x)

