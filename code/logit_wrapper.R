###### Variable selection for the Logit model. A wrapper approach #####

# A model building approach with sequential forward selection is 
# established in order to find the best subset of features. 
# Each model is built with crossvalidation on 
# the AUC measure.

set.seed(2610) 

logit_task_wrapper    <- makeClassifTask(data = train[, -1], target = "customer", 
                                         positive = "good")

logit_learner_wrapper <- makeLearner("classif.logreg", 
                                     predict.type = "prob") 

logit_ctrl_wrapper    <- makeFeatSelControlSequential(method = "sfs", 
                                                      alpha = 0.00001)
 
logit_rdesc_wrapper   <- makeResampleDesc("CV", iters = 3)

logit_sfeats          <- selectFeatures(learner = logit_learner_wrapper,
                                        task = logit_task_wrapper,
                                        resampling = logit_rdesc_wrapper,
                                        control = logit_ctrl_wrapper,
                                        show.info = TRUE,
                                        measures = mlr::auc)

# Performance score for each combination
analyzeFeatSelResult(logit_sfeats)

# Next, I store the optimal set of features to later use it 
# in the model building part file "logit_model.R"
vars$logit <- c("customer", logit_sfeats$x)

