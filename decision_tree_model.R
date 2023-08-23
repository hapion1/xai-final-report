library(tidymodels)
library(modeldata)

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
dt_tune_model <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
  ) %>% 
    set_engine("rpart") %>% 
    set_mode("classification")

# k-fold cross validation
# create folds
attrition_folds <- vfold_cv(
  attrition_train,
  v = 5,
  strata = Attrition
)

# Define metrics function 
# TODO: improve, add more, verify (metrics must be compatible, see https://yardstick.tidymodels.org/reference/metric_set.html)
attrition_metrics <- metric_set(roc_auc, accuracy, f_meas, precision, sens)

# Define workflow
attrition_dt_workflow <- workflow() %>% 
  add_model(dt_tune_model) %>% 
  add_recipe(attrition_recipe)

# Hyperparameter tuning with grid search
dt_grid <- grid_random(
  parameters(dt_tune_model),
  size = 3  # increase when finished for optimal params
)

dt_tuning <- attrition_dt_workflow %>% 
  tune_grid(
    resamples = attrition_folds,
    grid = dt_grid,
    metrics = attrition_metrics
  )

# Show tuning results
dt_tuning_results <- dt_tuning %>% 
  collect_metrics(summarize = FALSE)
dt_tuning_results

# Explore detailed ROC AUC for each fold
dt_tuning_results %>%
  filter(.metric == "roc_auc") %>%
  group_by(id) %>%
  summarize(
    min_roc_auc = min(.estimate),
    median_roc_auc = median(.estimate),
    max_roc_auc = max(.estimate)
  )

# Display 5 best performing models
dt_tuning %>%
  show_best(metric = 'roc_auc', n = 5)

# Automatic selection of best model
best_dt_model <- dt_tuning %>%
  # Choose the best model based on roc_auc
  select_best(metric = "roc_auc")

# Finalize workflow
final_attrition_workflow <- attrition_dt_workflow %>% 
  finalize_workflow(best_dt_model)
# final_attrition_workflow

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


