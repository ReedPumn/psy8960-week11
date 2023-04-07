## Script Settings and Resources
library(tidyverse)
library(haven)
library(caret)
library(tictoc)
library(parallel)
library(doParallel)
set.seed(123)

## Data Import and Cleaning
gss_tbl <- read_spss("GSS2016.sav") %>%
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
OLStocparallel <- toc()
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
# Note: I would have removed the leading zeros to Table 3, but I realized that after running my data with the supercomputer. I chose not to rerun this entire batch to remove leading zeros to save electricity and computational resources.
# Column headings in Table 3 and Table 4 were updated to reflect that the supercomputer produced these results.
Table3 <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  supercomputer = c(FirstR2, SecondR2, ThirdR2, FourthR2),
  supercomputer-12 = c(holdout1, holdout2, holdout3, holdout4)
)

# Column names were updated to reflect the number of cores used during processing with the supercomputer.
Table4 <- tibble(algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
                supercomputer = c(OLStoc$callback_msg, Enettoc$callback_msg, random_forest_gumptoc$callback_msg, EGBtoc$callback_msg),
                "supercomputer-12" = c(OLStocparallel$callback_msg, Enettocparallel$callback_msg, random_forest_gumptocparallel$callback_msg, EGBtocparallel$callback_msg))

# These lines export Table3 and Table4 to a csv format.
write_csv(Table3, "table3.csv")
write_csv(Table4, "table4.csv")

# Q1: Which models benefited most from moving to the supercomputer and why?
# A: The eXtreme Gradient Boosted model most benefited most from moving to the supercomputer because it reduced the largest number of seconds from the run time. It also had the largest percentage in reduced seconds of run time. As a result, it performed best under several metrics. This was by far the biggest improvement across models.

# Q2: What is the relationship between time and the number of cores used?
# A: There is a negative relationship between time and the number of cores used. As we allocate more cores to computing, run time decreases due to parallelization.

# Q3: If your supervisor asked you to pick a model for use in a production model, would you recommend using the supercomputer and why? Consider all four tables when providing an answer.
# A: For this question, I will assume that I am not going to change the models further and am only motivated to minimize cost. After all, I could always tune the models further to make them consider more hyperparameters for lower cost at the sacrifice of run time. Given that assumption, I would choose the Random Forest model because it produced the highest R-squared value when predicting data in the test set. It was not a close comparison, either; the Random Forest model had 9% more variance explained in the test data as compared to the next best model. If multiple models had near equivalence in their cost, then the argument could be made that further testing across different hyperparameters should be conducted. But because there was no near equivalence between models in terms of the test datas' R-squared values, I find the Random Forest model to be a clear winner. There are certainly tradeoffs to this decision with regards to communicability. The OLS Regression model, for instance, is much more communicable than the Random Forest model. As a result, it would take thoughtful conversations in an applied setting to help communicate what the model is doing and why I chose it over a model that could be more easily understood. Run time is also a consideration. If the models would be ran several times or if we needed to get results as fast as possible, then I might recommend the Elastic Net model since it also produced a high R-squared value in the holdout sample but while taking only a few seconds to run with supercomputing.