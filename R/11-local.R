## Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(tictoc)
library(parallel)
library(doParallel)
set.seed(123)

## Data Import and Cleaning
gss_tbl <- read_spss("../data/GSS2016.sav") %>%
  # Remove the other two columns that predict a similar criterion.
  select(-HRS1, -HRS2) %>%
  filter(!is.na(MOSTHRS)) %>%
  sapply(as.numeric) %>%
  as_tibble() %>%
  select_if(~(mean(is.na(.)) * 100) <= 75)

## Visualization
(ggplot(gss_tbl, aes(x = MOSTHRS)) +
    geom_histogram(binwidth = 10) +
    labs(x = "Workhours", y = "Count")) %>%
  ggsave("../figs/fig1.png", ., width = 1920, height = 1080, units = "px")

## Analysis
gss_random_tbl <- gss_tbl[sample(nrow(gss_tbl)), ]
gss_random75 <- round(nrow(gss_random_tbl) * 0.75, 0)
gss_train_tbl <- gss_random_tbl[1:gss_random75, ]
kfolds <- createFolds(gss_train_tbl$MOSTHRS, 10)
gss_test_tbl <- gss_random_tbl[(gss_random75 + 1):nrow(gss_random_tbl), ]

# The tic() toc() functions tell us how much time passed to execute all code around which they are wrapped. They are used to see how efficient our run time is. Importantly, we save the amount of time they calculate to an object to be used later in our table2_tbl. tic() and toc() are used frequently throughout this code, but only explained in a comment here for the sake of simplicity. 
tic()
OLS <- train(
  MOSTHRS ~ .,
  gss_train_tbl,
  method = "lm",
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
OLStoc <- toc()
OLS

tic()
Enet <- train(
  MOSTHRS ~ .,
  gss_train_tbl,
  method = "glmnet",
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
Enettoc <- toc()
Enet

tic()
random_forest_gump <- train(
  MOSTHRS ~ .,
  gss_train_tbl,
  method = "ranger",
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE))
random_forest_gumptoc <- toc()
random_forest_gump

tic()
EGB <- train(
  MOSTHRS ~ .,
  gss_train_tbl,
  method = "xgbLinear", 
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE))
EGBtoc <- toc()
EGB

# Now we attempt to run the same models, but this time with parallelized processing. My computer has 8 clusters, so the following two lines divide the processing demands across these clusters. The models are almost identical, so they will not be described in comments for the sake of simplicity. The only differences are that I renamed the models and toc() output to a different name to enable comparisons in the table2_tbl. All other aspects of the models are identical.
num_clusters <- makeCluster(7)
registerDoParallel(num_clusters)

tic()
OLSparallel <- train(
  MOSTHRS ~ .,
  gss_train_tbl,
  method = "lm",
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
OLSparalleltoc <- toc()
OLSparallel

tic()
Enetparallel <- train(
  MOSTHRS ~ .,
  gss_train_tbl,
  method = "glmnet",
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
Enettocparallel <- toc()
Enetparallel

tic()
random_forest_gumpparallel <- train(
  MOSTHRS ~ .,
  gss_train_tbl,
  method = "ranger",
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE))
random_forest_gumptocparallel <- toc()
random_forest_gumpparallel

tic()
EGBparallel <- train(
  MOSTHRS ~ .,
  gss_train_tbl,
  method = "xgbLinear", 
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE))
EGBtocparallel <- toc()
EGBparallel

# These two lines of code stop the parallelized processing. We return to normal processing because the computational load of running these models is done.
stopCluster(num_clusters)
registerDoSEQ()

FirstR2 <- OLS$results$Rsquared %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
SecondR2 <- max(Enet$results$Rsquared) %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
ThirdR2 <- max(random_forest_gump$results$Rsquared) %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
FourthR2 <- max(EGB$results$Rsquared) %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")

holdout1 <- cor(predict(OLS, gss_test_tbl, na.action = na.pass), gss_test_tbl$MOSTHRS)^2  %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0") 
holdout2 <- cor(predict(Enet, gss_test_tbl, na.action = na.pass), gss_test_tbl$MOSTHRS)^2  %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
holdout3 <- cor(predict(random_forest_gump, gss_test_tbl, na.action = na.pass), gss_test_tbl$MOSTHRS)^2  %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
holdout4 <- cor(predict(EGB, gss_test_tbl, na.action = na.pass), gss_test_tbl$MOSTHRS)^2 %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")

## Publication
table1_tbl <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  cv_rsq = c(FirstR2, SecondR2, ThirdR2, FourthR2),
  ho_rsq = c(holdout1, holdout2, holdout3, holdout4)
)

# This series of pipes creates a new tibble that outlines the run times of our various models when executed either normally or with parallelization. We do this to compare efficiency increases in run time.
table2_tbl <- tibble(algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
                original = c(OLStoc$callback_msg, Enettoc$callback_msg, random_forest_gumptoc$callback_msg, EGBtoc$callback_msg),
                parallelized = c(OLSparalleltoc$callback_msg, Enettocparallel$callback_msg, random_forest_gumptocparallel$callback_msg, EGBtocparallel$callback_msg))

# Q1: Which models benefited most from parallelization and why?
# A: 

# Q2: How big was the difference between the fastest and slowest parallelized model? Why?
# A: 

# Q3: If your supervisor asked you to pick a model for use in a production model, which would you recommend and why? Consider both Table 1 and Table 2 when providing an answer.
# A: 