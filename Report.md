# Machine Learning Prediction Assignment
# Executive Summary
IN this project I develop a model to predict the manner in which the exercise is done by using the data collected by Wallace Ugulino, Eduardo Velloso, and Hugo Fuks.The data was collected as  part of a project on Human Activity Recognition experiment.

# Setting up the R environment
Firs R environment was set up by calling all the the required libraries.

```r
suppressWarnings(suppressMessages(library(rpart)))
suppressWarnings(suppressMessages(library(caret)))
suppressWarnings(suppressMessages(library(ggplot2)))
suppressWarnings(suppressMessages(library(ipred)))
suppressWarnings(suppressMessages(library(randomForest)))
suppressWarnings(suppressMessages(library(rpart.plot)))
```

# Reading and cleaning up the data
The data set is collected for 6 participants while performing one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions. The training data set collect 19622 observations with 160 fields, the outcome is named "classe". The test data set collect 20 observations with 160 fields, the outcome is missed. The outcome of this dataset will be predicted by the model created and submitted for evaluation. To clean up the data, I first removed the columns that contained NA and then removed few columns of the dataset that do not contributed to the accelerometer measurements. The following code is used to read and clean up the data:

```r
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(url_train, method="libcurl"), na.strings=c("NA","#DIV/0!",""))
testing  <- read.csv(url(url_test, method="libcurl"), na.strings=c("NA","#DIV/0!",""))
NAcond <- (colSums(is.na(training)) == 0)
training <- training[, NAcond]
testing <- testing[, NAcond]
regex <- grepl("^X|timestamp|user_name", names(training))
training <- training[, !regex]
testing <- testing[, !regex]
```
# Partitioning Training Set
Now I split the cleaned training set into a pure training data set (60%) and a validation data set (40%). We will use the validation data set to conduct cross validation in future steps.The entire dataset now divided into three sections as: (1) training, (2) validation and (3) testing dataset. 

```r
set.seed(2) # For reproducibile purpose
inTrain <- createDataPartition(training$classe, p = 0.60, list = FALSE)
validation <- training[-inTrain, ]
training <- training[inTrain, ]
```
#Data Modeling
##Decision Tree
We fit a predictive model for activity recognition using Decision Tree algorithm. Output is shown in <b>Appendix 1</b>. Then I estimate the performance of the model on the validation data set.

```r
modelTree <- rpart(classe ~ ., data = training, method = "class")
predictTree <- predict(modelTree, validation, type = "class")
accuracy <- postResample(predictTree, validation$classe)
ose <- 1 - as.numeric(confusionMatrix(validation$classe, predictTree)$overall[1])
```
74.6367576% and 25.3632424% are the estimated accuracy of the Random Forest Model and the estimated Out-of-Sample Error, respectively.

## Random Forest
We fit a predictive model for activity recognition using <b>Random Forest</b> algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. model output is shown in <b>Appendix 2</b>. I then use <b>5-fold cross validation</b> when applying the algorithm. I estimate the performance of the model on the <b>validation</b> data set.  

```r
modelRF <- train(classe ~ ., data = training, method = "rf", 
                 trControl = trainControl(method = "cv", 5), ntree = 100)
predictRF <- predict(modelRF, validation)
accuracy <- postResample(predictRF, validation$classe)
ose <- 1 - as.numeric(confusionMatrix(validation$classe, predictRF)$overall[1])
```
The Estimated Accuracy of the Random Forest Model is 99.5794035% and the Estimated Out-of-Sample Error is 0.4205965%. Random Forests yielded better Results than decision three and therefore is considered as final model!

## Make Predictions for Test Data Set  
Now, we apply the <b>Random Forest</b> model to the original testing data set downloaded from the data source. We remove the problem_id column first.  

```r
predict(modelRF, testing[, -length(names(testing))])
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
#Appendix
###Appendix 1: Decision Tree Output and Confusion marix of the model on the <b>validation</b> data set is below.

```r
prp(modelTree)
```

![](Report_files/figure-html/unnamed-chunk-7-1.png)

```r
confusionMatrix(validation$classe, predictRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    7 1509    2    0    0
##          C    0    8 1360    0    0
##          D    0    0    9 1277    0
##          E    0    0    0    7 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9958          
##                  95% CI : (0.9941, 0.9971)
##     No Information Rate : 0.2854          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9947          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9969   0.9947   0.9920   0.9945   1.0000
## Specificity            1.0000   0.9986   0.9988   0.9986   0.9989
## Pos Pred Value         1.0000   0.9941   0.9942   0.9930   0.9951
## Neg Pred Value         0.9988   0.9987   0.9983   0.9989   1.0000
## Prevalence             0.2854   0.1933   0.1747   0.1637   0.1829
## Detection Rate         0.2845   0.1923   0.1733   0.1628   0.1829
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9984   0.9967   0.9954   0.9966   0.9995
```
###Appendix 2: Random Forest Output

```r
modelRF
```

```
## Random Forest 
## 
## 11776 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9420, 9421, 9421, 9421 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa    
##    2    0.990744  0.9882913
##   28    0.994735  0.9933406
##   54    0.992527  0.9905482
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```


