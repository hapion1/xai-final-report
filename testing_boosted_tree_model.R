library(tidymodels)
library(modeldata)
library(xgboost)

# trees with k-fold cross validation and hyperparameter tuning
# Split dataset
set.seed(42)
attrition_split <- initial_split(
  attrition,
  prop = 0.8,
  strata = Attrition
)

attrition_train <- attrition_split %>% training()
attrition_test <- attrition_split %>% testing()

# Preprocessing: drop correlated, normalize, dummy encoding
# create recipe
attrition_recipe <- recipe(
  Attrition ~ .,
  data = attrition_train
) %>% 
  step_corr(all_numeric(), threshold = 0.7) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), -all_outcomes())

# Create model that will be tuned
boost_tune_model <- boost_tree(
  # mtry = tune(),  # does not work with tune_grid()
  trees = tune(),  # maybe manually
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = tune()
) %>% 
  # set_engine("xgboost") %>% 
  set_engine("xgboost", nthread=6) %>%  # CHECK OUT MULTIPROCESSING!
  set_mode("classification")

# k-fold cross validation
# create folds
attrition_folds <- vfold_cv(
  attrition_train,
  v = 5,
  strata = Attrition
)

# Create tuning grid
tunegrid_boost <- grid_regular(
  parameters(boost_tune_model),
  levels = 2
)

# Tune along the grid
tune_results <- tune_grid(
  boost_tune_model,
  Attrition ~ .,
  resamples = attrition_folds,
  grid = tunegrid_boost,
  metrics = metric_set(roc_auc)
)

# Plot results
autoplot(tune_results)

# Select best hyperparameters
best_params <- select_best(tune_results)

# Finalize the model specification
final_spec <- finalize_model(boost_tune_model, best_params)

# Train the model
final_boost_model <- final_spec %>% 
  fit(
    Attrition ~ .,
    data = attrition_train
  )

# Predict BROKEN needs fix
pred_test <- predict(final_boost_model, new_data = attrition_test, type = "prob") %>% 
  bind_cols(attrition_test)

auc_boost <- roc_auc(pred_test, truth = Attrition, estimate = .pred_No)











# Define metrics function 
# TODO: improve, add more, verify (metrics must be compatible, see https://yardstick.tidymodels.org/reference/metric_set.html)
attrition_metrics <- metric_set(roc_auc, accuracy, f_meas, precision, sens)

# Define workflow
attrition_boost_workflow <- workflow() %>% 
  add_model(boost_tune_model) %>% 
  add_recipe(attrition_recipe)

# Hyperparameter tuning with grid search
boost_grid <- grid_random(
  parameters(boost_tune_model),
  size = 3  # increase when finished for optimal params
)

boost_tuning <- attrition_dt_workflow %>% 
  tune_grid(
    resamples = attrition_folds,
    grid = boost_grid,
    metrics = attrition_metrics
  )

# Show tuning results
boost_tuning_results <- boost_tuning %>% 
  collect_metrics(summarize = FALSE)
boost_tuning_results

# Explore detailed ROC AUC for each fold
boost_tuning_results %>%
  filter(.metric == "roc_auc") %>%
  group_by(id) %>%
  summarize(
    min_roc_auc = min(.estimate),
    median_roc_auc = median(.estimate),
    max_roc_auc = max(.estimate)
  )

# Display 5 best performing models
boost_tuning %>%
  show_best(metric = 'roc_auc', n = 5)

# Automatic selection of best model
best_boost_model <- boost_tuning %>%
  # Choose the best model based on roc_auc
  select_best(metric = "roc_auc")

# Finalize workflow
final_attrition_workflow <- attrition_boost_workflow %>% 
  finalize_workflow(best_boost_model)
# final_attrition_workflow


############ ab hier keine var umbenannt ##############
# Train model
attrition_final_fit <- final_attrition_workflow %>% 
  last_fit(split = attrition_split)

# View performance metrics
attrition_final_fit %>% 
  collect_metrics()

# Create ROC curve
attrition_final_fit %>% 
  collect_predictions() %>% 
  roc_curve(truth = Attrition, .pred_No) %>% 
  autoplot()

# Predictions
predictions <- attrition_final_fit %>% 
  collect_predictions()

# Calculate roc_auc
metric_roc_auc <- roc_auc(
  predictions,
  truth = Attrition,
  .pred_No
)
# metric_roc_auc

# Create confusion matrix
confusion <- predictions %>% 
  select(-id, -.row)

conf_mat(
  confusion,
  truth = Attrition,
  estimate = .pred_class
) %>% 
  autoplot("mosaic")
# autoplot("heat)


