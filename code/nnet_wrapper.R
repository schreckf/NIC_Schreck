###### Feature selection for the Neural Network. A wrapper approach #######

# A model building approach with sequential forward selection is 
# established in order to find the best subset of features. 
# Each model is built with crossvalidation on the AUC measure.

set.seed(2610) 

nnet_task_wrapper     <- makeClassifTask(data = train, 
                                         target = "customer", 
                                         positive = "good")

nnet_learner_wrapper  <- makeLearner("classif.nnet",
                                     predict.type = "prob",
                                     "trace" = FALSE)

nnet_ctrl_wrapper     <- makeFeatSelControlSequential(method = "sfs", 
                                                      alpha = 0.00001)

nnet_rdesc_wrapper    <- makeResampleDesc("CV", iters = 3)

nnet_sfeats           <- selectFeatures(learner = nnet_learner_wrapper,
                                        task = nnet_task_wrapper, 
                                        resampling = nnet_rdesc_wrapper,
                                        control = nnet_ctrl_wrapper, 
                                        show.info = TRUE,
                                        measures = mlr::auc)

# Performance score for each combination
analyzeFeatSelResult(nnet_sfeats)

# Next, I store the optimal set of features to later use it 
# in the model building part file "nnet_model.R"
vars$nnet <- c("customer", nnet_sfeats$x)

