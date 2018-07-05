###### Variable selection for the Decision Tree model. A wrapper approach #####

# A model building approach with sequential forward selection is 
# established in order to find the best subset of features. 
# Each model is built with crossvalidation on the AUC measure.

set.seed(2610) 

dt_task_wrapper     <- makeClassifTask(data = train, target = "customer", 
                                       positive = "good")

dt_learner_wrapper  <- makeLearner("classif.rpart",
                                   predict.type = "prob") 

# We set the minimum required difference to 0.0001 Euros in expected loss
# per observation
dt_ctrl_wrapper     <- makeFeatSelControlSequential(method = "sfs", 
                                                    alpha = 0.00001) 

dt_rdesc_wrapper    <- makeResampleDesc("CV", iters = 3)
 
dt_sfeats           <- selectFeatures(learner = dt_learner_wrapper,
                                      task = dt_task_wrapper,
                                      resampling = dt_rdesc_wrapper,
                                      control = dt_ctrl_wrapper,
                                      show.info = TRUE,
                                      measures = mlr::auc)

# Performance score for each combination of features
analyzeFeatSelResult(dt_sfeats)

# Next, I store the optimal set of features to later use it 
# in the model building part file "dt_model.R"
vars$dt <- c("customer", dt_sfeats$x)

