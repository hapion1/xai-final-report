library(tidyverse)
library(tidymodels)
library(parsnip)
library(recipes)
library(yardstick)
library(tune) # for tuning functions
library(themis)
library(xgboost)
library(doParallel)
library(ggplot2)


set.seed(42)

grid_size = 500
vfolds = 5

# Function to do preprocessing
prep_data <- function(data) {
  rec <- recipe(Attrition ~ ., data = data) %>%
    # Remove highly correlated numeric features
    step_corr(all_numeric(), threshold = 0.8) %>%
    # Remove zero variance predictors
    step_zv(all_predictors()) %>%
    # Convert nominal features to dummy variables
    step_dummy(all_nominal(), -all_outcomes()) %>%
    # Normalize predictors
    step_normalize(all_predictors()) %>%
    # Upsample the minority class using SMOTE
    step_smote(Attrition) %>%
    # Center and scale numeric features
    step_center(all_numeric(), -all_outcomes()) %>%
    # Scale numeric features
    step_scale(all_numeric(), -all_outcomes())
  return(rec)
}

# Function to tune model
tune_model <- function(model, data, metrics, param_grid, vfolds) {
  workflow <- workflow() %>%
    add_model(model) %>%
    add_recipe(prep_data(data))
  
  folds <- vfold_cv(data, v = vfolds, strata = Attrition)
  
  tuning <- workflow %>% 
    tune_grid(
      resamples = folds,
      grid = param_grid,
      metrics = metrics,
      control = control_grid(verbose = TRUE)
    )
  return(tuning)
}

# Function to finalize workflow and fit model
final_fit <- function(model_spec, data) {
  
  workflow <- workflow() %>%
    add_model(model_spec) %>% 
    add_recipe(prep_data(data))
  
  fit <- workflow %>%
    last_fit(split = data, metrics = yardstick::metric_set(roc_auc, yardstick::accuracy,yardstick::precision, yardstick::recall, yardstick::f_meas) )
  
  return(fit)
}

# Load data 
data("attrition")

data_split <- initial_split(attrition, prop = 0.8, strata = Attrition)

train_data <- data_split %>% training()
test_data <- data_split %>% testing()


# XGBoost model
xgb_model_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune()
) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

# Define initial parameter ranges
param_grid <- parameters(xgb_model_spec)

# Update parameter ranges
wparam_grid <- param_grid %>%
  update(trees = trees(range = c(1, 30))) %>%
  update(tree_depth = tree_depth(range = c(1, 10))) %>%
  update(min_n = min_n(range = c(1, 20))) %>%
  update(loss_reduction = loss_reduction(range = c(0, 1)))

# Generate a regular grid  
xgb_param_grid <- grid_random(param_grid, size = grid_size)

# Random Forest model
rf_model <- rand_forest(
  trees = tune(),
  #mtry = tune(),
  min_n = tune()  
) %>%
  set_mode("classification")

# Define initial parameter ranges for Random Forest
rf_param_grid <- parameters(rf_model) 

# Update parameter ranges for Random Forest
#rf_param_grid <- rf_param_grid %>%
#update(trees = trees(range = c(1, 29))) %>%
#update(mtry = mtry(range = c(1, 10))) %>%
#update(min_n = min_n(range = c(1, 20)))

# Generate a regular grid
rf_param_grid <- grid_random(rf_param_grid, size = grid_size)

# SVM model
svm_model <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Define initial parameter ranges for SVM
svm_param_grid <- parameters(svm_model)

# Update parameter ranges for SVM
wsvm_param_grid <- svm_param_grid %>%
  update(cost = cost(range = c(1, 10))) %>%
  update(rbf_sigma = rbf_sigma(range = c(0.1, 1)))

# Generate a regular grid
svm_param_grid <- grid_random(svm_param_grid, size = grid_size)

# Logistic Regression model
logreg_model <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# Define initial parameter ranges for Logistic Regression
logreg_param_grid <- parameters(logreg_model)

# Update parameter ranges for Logistic Regression
wlogreg_param_grid <- logreg_param_grid %>%
  update(penalty = penalty(range = c(0, 1))) %>%
  update(mixture = mixture(range = c(0, 1)))

# Generate a regular grid
logreg_param_grid <- grid_random(logreg_param_grid, size = grid_size)

# Decision Tree model
dt_model <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("rpart")

# Define initial parameter ranges for Decision Tree
dt_param_grid <- parameters(dt_model)

# Update parameter ranges for Decision Tree
wdt_param_grid <- dt_param_grid %>%
  update(cost_complexity = cost_complexity(range = c(0.000000001, 0.0001))) %>%
  update(tree_depth = tree_depth(range = c(1, 10))) %>%
  update(min_n = min_n(range = c(1, 10)))

# Generate a regular grid
dt_param_grid <- grid_random(dt_param_grid, size = grid_size)

# Define metrics
metrics <- yardstick::metric_set(roc_auc, yardstick::accuracy, 
                                 yardstick::precision, yardstick::recall, yardstick::f_meas)
metrics

# Tune XGBoost model
print("Tune XGBoost model")
xgb_tuning <- tune_model(xgb_model_spec, train_data, metrics, xgb_param_grid, vfolds)
# Tune SVM model
print("Tune SVM model")
svm_tuning <- tune_model(svm_model, train_data, metrics, svm_param_grid, vfolds)
# Tune Random Forest model 
print("Tune Random Forest model")
rf_tuning <- tune_model(rf_model, train_data, metrics, rf_param_grid, vfolds)
# Tune Logistic Regression model 
print("Tune Logistic Regression model")
logreg_tuning <- tune_model(logreg_model, train_data, metrics, logreg_param_grid, vfolds)
# Tune Decision Tree model
print("Tune Decision Tree model")
dt_tuning <- tune_model(dt_model, train_data, metrics, dt_param_grid, vfolds)

# Get best XGBoost model spec
best_xgb <- xgb_tuning %>% 
  select_best(metric = "roc_auc")

# Get best Random Forest model spec
best_rf <- rf_tuning %>%
  select_best(metric = "roc_auc")

# Get best SVM model spec
best_svm <- svm_tuning %>%
  select_best(metric = "roc_auc")

# Get best Logistic Regression model spec
best_logreg <- logreg_tuning %>%
  select_best(metric = "roc_auc")

# Get best Decision Tree model spec
best_dt <- dt_tuning %>%
  select_best(metric = "roc_auc")

best_xgb_spec <- xgb_model_spec %>% 
  set_args(
    trees = best_xgb$trees,
    tree_depth = best_xgb$tree_depth,
    min_n = best_xgb$min_n,
    loss_reduction = best_xgb$loss_reduction
  )

best_svm_spec <- svm_model %>%
  set_args(
    cost = best_svm$cost,
    rbf_sigma = best_svm$rbf_sigma
  )

# Update the model spec with the best parameters
best_rf_spec <- rf_model %>% 
  set_args(
    trees = best_rf$trees,
    #mtry = best_rf$mtry,
    min_n = best_rf$min_n
  )

best_logreg_spec <- logreg_model %>% 
  set_args(
    penalty = best_logreg$penalty,
    mixture = best_logreg$mixture
  )

best_dt_spec <- dt_model %>% 
  set_args(
    cost_complexity = best_dt$cost_complexity,
    tree_depth = best_dt$tree_depth,
    min_n = best_dt$min_n
  )

# Finalize workflow and fit XGBoost model
xgb_fit <- final_fit(best_xgb_spec, data_split)

# Finalize workflow and fit Random Forest model
rf_fit <- final_fit(best_rf_spec, data_split)

# Finalize workflow and fit SVM model
svm_fit <- final_fit(best_svm_spec, data_split)

# Finalize workflow and fit Logistic Regression model
logreg_fit <- final_fit(best_logreg_spec, data_split)

# Finalize workflow and fit Decision Tree model
dt_fit <- final_fit(best_dt_spec, data_split)

xgb_fit %>% 
  collect_predictions() %>% 
  roc_curve(truth = Attrition, .pred_No) %>% 
  autoplot() +
  ggtitle("XGBoost ROC Curve")

rf_fit %>% 
  collect_predictions() %>% 
  roc_curve(truth = Attrition, .pred_No) %>% 
  autoplot() +
  ggtitle("Random Forest ROC Curve")

svm_fit %>% 
  collect_predictions() %>% 
  roc_curve(truth = Attrition, .pred_No) %>% 
  autoplot() +
  ggtitle("Support Vector Machine ROC Curve")

logreg_fit %>% 
  collect_predictions() %>% 
  roc_curve(truth = Attrition, .pred_No) %>% 
  autoplot() +
  ggtitle("Logistic Regression ROC Curve")

dt_fit %>% 
  collect_predictions() %>% 
  roc_curve(truth = Attrition, .pred_No) %>% 
  autoplot() +
  ggtitle("Decision Tree ROC Curve")

# Collect metrics
xgb_metrics <- xgb_fit %>% collect_metrics()
rf_metrics <- rf_fit %>% collect_metrics() 
svm_metrics <- svm_fit %>% collect_metrics()
logreg_metrics <- logreg_fit %>% collect_metrics()
dt_metrics <- dt_fit %>% collect_metrics()

# Define a function to append metrics to a text file
append_metrics_to_file <- function(metrics, model_name, file_path) {
  metrics_string <- paste(
    model_name,
    "Metrics:\n",
    "----------------------------------------\n"
  )
  for (i in seq_along(metrics$.metric)) {
    metric_name <- metrics$.metric[i]
    estimator <- metrics$.estimator[i]
    estimate <- metrics$.estimate[i]
    
    metric_line <- sprintf("%s (%s): %f\n", metric_name, estimator, estimate)
    metrics_string <- paste(metrics_string, metric_line)
  }
  
  # Append metrics to the file (use 'append = TRUE')
  cat(metrics_string, "\n\n", file = file_path, append = TRUE)
}

# Define the file path
metrics_file_path <- "metrics_results.txt"

# Append metrics to the text file for each model
append_metrics_to_file(xgb_metrics, "XGBoost", metrics_file_path)
append_metrics_to_file(rf_metrics, "Random Forest", metrics_file_path)
append_metrics_to_file(svm_metrics, "SVM", metrics_file_path)
append_metrics_to_file(logreg_metrics, "Logistic Regression", metrics_file_path)
append_metrics_to_file(dt_metrics, "Decision Tree", metrics_file_path)

# Print metrics
print("XGBoost Metrics:")
xgb_metrics %>%
  select(.metric, .estimator, .estimate)

print("Random Forest Metrics:")
rf_metrics %>%
  select(.metric, .estimator, .estimate)

print("SVM Metrics:")
svm_metrics %>%
  select(.metric, .estimator, .estimate)

print("Logistic Regression Metrics:")
logreg_metrics %>%
  select(.metric, .estimator, .estimate)

print("Decision Tree Metrics:")
dt_metrics %>%
  select(.metric, .estimator, .estimate)


xgb_tuning %>%
  show_best(metric = 'roc_auc', n = 5)

rf_tuning %>%
  show_best(metric = 'roc_auc', n = 5)

svm_tuning %>%
  show_best(metric = 'roc_auc', n = 5)

logreg_tuning %>%
  show_best(metric = 'roc_auc', n = 5)

dt_tuning %>%
  show_best(metric = 'roc_auc', n = 5)