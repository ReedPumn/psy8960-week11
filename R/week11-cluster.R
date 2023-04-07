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
# Note: I would have removed the leading zeros to Table 3 and Table 4, but I realized that after running my data with the supercomputer. I chose not to rerun this to remove leading zeros to save electricity and computational resources.
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

# 