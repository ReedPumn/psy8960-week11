## Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
set.seed(123)

## Data Import and Cleaning
gss_tbl <- read_spss("../data/GSS2016.sav") %>%
  filter(!is.na(HRS1)) %>%
  sapply(as.numeric) %>%
  as_tibble() %>%
  select_if(~(mean(is.na(.)) * 100) <= 75)

## Visualization
(ggplot(gss_tbl, aes(x = HRS1)) +
    geom_histogram(binwidth = 10) +
    labs(x = "Workhours", y = "Count")) %>%
  ggsave("../figs/fig1.png", ., width = 1920, height = 1080, units = "px")

## Analysis
gss_random_tbl <- gss_tbl[sample(nrow(gss_tbl)), ]
gss_random75 <- round(nrow(gss_random_tbl) * 0.75, 0)
gss_train_tbl <- gss_random_tbl[1:gss_random75, ]
kfolds <- createFolds(gss_train_tbl$HRS1, 10)
gss_test_tbl <- gss_random_tbl[(gss_random75 + 1):nrow(gss_random_tbl), ]

OLS <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "lm",
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
OLS

Enet <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "glmnet",
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE)
)
Enet

random_forest_gump <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "ranger",
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE))
random_forest_gump

EGB <- train(
  HRS1 ~ .,
  gss_train_tbl,
  method = "xgbLinear", 
  na.action = na.pass,
  preProcess = c("center", "scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", indexOut = kfolds, number = 10, search = "grid", verboseIter = TRUE))
EGB

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

holdout1 <- cor(predict(OLS, gss_test_tbl, na.action = na.pass), gss_test_tbl$HRS1)^2  %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0") 
holdout2 <- cor(predict(Enet, gss_test_tbl, na.action = na.pass), gss_test_tbl$HRS1)^2  %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
holdout3 <- cor(predict(random_forest_gump, gss_test_tbl, na.action = na.pass), gss_test_tbl$HRS1)^2  %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")
holdout4 <- cor(predict(EGB, gss_test_tbl, na.action = na.pass), gss_test_tbl$HRS1)^2 %>%
  round(2) %>%
  str_remove(pattern = "^(?-)0")

## Publication

table1_tbl <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  cv_rsq = c(FirstR2, SecondR2, ThirdR2, FourthR2),
  ho_rsq = c(holdout1, holdout2, holdout3, holdout4)
)
