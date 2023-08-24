library(tidymodels)
library(modeldata)
library(xgboost)
library(doParallel)

# Setup multiprocessing (especially for faster tuning)
all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)


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
  set_engine("xgboost") %>% 
  # set_engine("xgboost", nthread=6) %>%  # CHECK OUT MULTIPROCESSING!
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

