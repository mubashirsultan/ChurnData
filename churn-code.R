# ChurnData
load('/Users/mubashirsultan/Documents/R/Week7/churnData.RData')

View(churnData)

churn <- churnData[,20]
dummyInfo <- dummyVars(~ . , data=churnData[,-20])
churnData <- predict(dummyInfo, churnData[,-20])
churnData <- data.frame(churnData)
churnData$churn <- churn

set.seed(1)
trainIndex <- createDataPartition(churnData$churn, p=0.8, list=FALSE)
head(trainIndex)

trainData <- churnData[trainIndex,]
testData  <- churnData[-trainIndex,]


##  Tune various models and compare predictive accuracy.

library(snow)
library(doSNOW)

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10, 
  classProbs = TRUE,
  summaryFunction = twoClassSummary)

#############################################

##  CART

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

rpartOut <- train(trainData[,-grep('churn', names(trainData))],
                  trainData[,grep('churn', names(trainData))],
                  method = "rpart", metric='ROC',
                  trControl = fitControl)
stopCluster(cl)

save(rpartOut, file='rpartOut.RData')
load('rpartOut.RData')
rpartPred <- predict(rpartOut, testData)

#############################################

##  Bagged CART

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

treeBagOut <- train(trainData[,-grep('churn', names(trainData))],
                    trainData[,grep('churn', names(trainData))],
                    method = "treebag",  metric='ROC',
                    trControl = fitControl)
stopCluster(cl)

save(treeBagOut, file='treeBagOut.RData')
load('treeBagOut.RData')

treeBagPred <- predict(treeBagOut, testData)

#############################################

#  Random Forest

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#fitControl2 <- trainControl(
#  method = "oob",
#  classProbs = TRUE,
#  summaryFunction = twoClassSummary)

rfOut <- train(trainData[,-grep('churn', names(trainData))],
               trainData[,grep('churn', names(trainData))],
               method = "rf",  metric='ROC',
               trControl = fitControl)
stopCluster(cl)

save(rfOut, file='rfOut.RData')
load('rfOut.RData')
rfPred <- predict(rfOut, testData)

#############################################

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)


set.seed(1)
gbmOut <- train(trainData[,-grep('churn', names(trainData))],
                trainData[,grep('churn', names(trainData))],
                method = "gbm",  metric='ROC', 
                trControl = fitControl)
stopCluster(cl)

save(gbmOut, file='gbmOut.RData')
load('gbmOut.RData')
gbmPred <- predict(gbmOut, testData)


#############################################

postResample(rpartPred, testData$churn)
postResample(treeBagPred, testData$churn)
postResample(rfPred, testData$churn)
postResample(gbmPred, testData$churn)

rpartProb <- predict(rpartOut, testData, type='prob')

treeBagProb <- predict(treeBagOut, testData, type='prob')

rfProb <- predict(rfOut, testData, type='prob')

gbmProb <- predict(gbmOut, testData, type='prob')

library(AUC)
rpartAuc <- auc(roc(rpartProb$yes, testData$churn))
treeBagAuc <- auc(roc(treeBagProb$yes, testData$churn))
rfAuc <- auc(roc(rfProb$yes, testData$churn))
gbmAuc <- auc(roc(gbmProb$yes, testData$churn))

barchart(c(rpart=rpartAuc, 
           treeBag=treeBagAuc,
           rf=rfAuc,
           gbm=gbmAuc),
         xlim=c(0,1))

plot(rpartOut)
plot(treeBagOut)
plot(rfOut)
plot(gbmOut)

