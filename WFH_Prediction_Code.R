# importing the dataset
data_ori <- read.csv("C:/Users/rajsh/Desktop/ML/Groupwork_2024_data/Data/Homeworkers.csv")
head(data_ori)
dim(data_ori)
data_ori$work_home <- as.factor(data_ori$work_home)
summary(data_ori)

# check for missing values
colSums(is.na(data_ori))
sum(data_ori == "")
data_ori[data_ori == ""] <- NA

# removing rows with missing values
data <- na.omit(data_ori)
colSums(is.na(data))

# checking for typographical errors
unique(data$salary_range)
unique(data$wfh_prev_workday)
unique(data$workday)

# looking for data outliers
boxplot(scale(data[,c(2,4,5,8)]))

# examining data features and removing outliers
data$distance_office[order(data$distance_office, decreasing = T) [1:10]]
data <- data[data$distance_office <= 20,]

data$gas_price[order(data$gas_price, decreasing = T) [1:10]]

data$public_transportation_cost[order(data$public_transportation_cost, decreasing = T) [1:10]]
data <- data[data$public_transportation_cost <= 10,]

data$tenure[order(data$tenure, decreasing = T) [1:10]]
data$tenure[order(data$tenure, decreasing = F) [1:5]]

# converting categorical variables to numerical variables

# creating data frame without class variable
transform <- data[,-9]
summary(transform)

# transforming the categorical variables to dummy variables
library(caret)
dmy <- dummyVars(" ~ .", data = transform, fullRank=T)
tranform_data <- data.frame(predict(dmy, newdata = transform))
summary(tranform_data)

# adding the class variable to the new data frame
tranform_data$work_home <- data$work_home
data <- tranform_data

#dropping the identifier column
data$identifier <- NULL

# checking multi collinearity of the predictors
plot(data[,c('distance_office','gas_price','public_transportation_cost','tenure')])
cor(data[,c('distance_office','gas_price','public_transportation_cost','tenure')])

# checking for balanced state
summary(data$work_home)
set.seed(12345)
data_down <- downSample(data[, -ncol(data)], data$work_home, yname = "work_home")
summary(data_down$work_home)
dim(data_down)

# normalizing the data
preProcValues <- preProcess(data_down, method = "range")
data_down <- predict(preProcValues, data_down)
summary(data_down)

# partitioning the data into 75% train and 25% test data
set.seed(12345)
index <- createDataPartition(data_down$work_home, p=0.75, list=FALSE)
training <- data_down[index,]
test <- data_down[-index,]
summary(training$work_home)
summary(test$work_home)

TControl <- trainControl(method="cv", number=10)

report <- data.frame(Model=character(), Acc.Train=numeric(), Acc.Test=numeric())


#**********************
# k-Nearest Neighbors
#**********************

set.seed(12345)
knnGrid <- expand.grid(k = c(seq(3, 25, 2)))
knnmodel <- train(work_home~., data=training, method="knn", tuneGrid=knnGrid, trControl=TControl)
knnmodel
prediction.train <- predict(knnmodel, training[,-13], type="raw")
prediction.test <- predict(knnmodel, test[,-13],type="raw")
acctr <- confusionMatrix(prediction.train, training[,13])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,13])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="k-NN", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

#**********************
#         C5.0
#**********************
set.seed(12345)
c50Grid <- expand.grid(trials = c(1:5),
                       model = c("tree", "rules"),
                       winnow = c(TRUE,FALSE))
c5model <- train(work_home ~., data=training, method="C5.0", tuneGrid=c50Grid, trControl=TControl)
c5model 
prediction.train <- predict(c5model, training[,-13], type="raw")
prediction.test <- predict(c5model, test[,-13], type="raw")
acctr <- confusionMatrix(prediction.train, training[,13])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,13])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="C5.0", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))



#**********************
#         CART
#**********************
set.seed(12345)
cart_grid <- expand.grid(cp = c(0.001,0.01,0.05,0.1))
cartmodel <- train(work_home ~., data=training, method="rpart", trControl=TControl, tuneGrid = cart_grid)
cartmodel
prediction.train <- predict(cartmodel, training[,-13], type="raw")
prediction.test <- predict(cartmodel, test[,-13], type="raw")
acctr <- confusionMatrix(prediction.train, training[,13])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,13])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="CART", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


# to visualize the trained cart model
# install.packages("rpart.plot") 
library(rpart.plot) # For plotting decision trees
pdf("CART.pdf", width = 12, height = 8)
rpart.plot(cartmodel$finalModel, 
           type=3, # Plot type (3 labels on both sides)
           extra=104, # Display probabilities and counts
           under=TRUE, # Show node counts below
           fallen.leaves = TRUE) # Align leaves for readibilty 
dev.off()

#**********************************
#          Random Forest
#**********************************

besta <- 0
bestn <- 0
bestm <- 0
besttree <- 0
for (maxnodes in c(10,20,30,50,70,85,100,150)) {
  for (n_tree in c(100,200,300,400,500,600,700)){
    set.seed(12345)
    rfGrid <- expand.grid(mtry = c(seq(2, 10)))
    rformodel <- train(work_home ~., data=training, method="rf", tuneGrid=rfGrid, trControl=TControl, maxnodes=maxnodes, ntree=n_tree)
    cat("\n\n maxnodes=", maxnodes, "  Accuracy=", max(rformodel$results$Accuracy), "  mtry=", rformodel$bestTune$mtry, " ntree=", n_tree)
    if (max(rformodel$results$Accuracy)>besta) {
      bestn <- maxnodes
      bestm <- rformodel$bestTune$mtry
      besta <- max(rformodel$results$Accuracy)
      besttree <- n_tree
    }
  }
} 

set.seed(12345)
rfGrid <- expand.grid(mtry = bestm)
rformodel <- train(work_home ~., data=training, method="rf", tuneGrid=rfGrid, trControl=TControl, maxnodes=bestn, ntree=besttree)
rformodel
prediction.train <- predict(rformodel, training[,-13], type="raw")
prediction.test <- predict(rformodel, test[,-13], type="raw")
acctr <- confusionMatrix(prediction.train, training[,13])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,13])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="Random Forest", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

#**********************
# Logistic Regression
#**********************
set.seed(12345)
lrmodel <- train(work_home~., data=training, method="glm", trControl=TControl)
lrmodel
prediction.train <- predict(lrmodel, training[,-13], type="raw")
prediction.test <- predict(lrmodel, test[,-13], type="raw")
acctr <- confusionMatrix(prediction.train, training[,13])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,13])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="Logistic Regression", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

#**********************
#   Neural Network
#**********************
set.seed(12345)
nn_grid <- expand.grid(
  size = c(1, 3, 5, 7, 9), decay = c(0, 0.001, 0.01, 0.1)  
)

nnmodel <- train(work_home ~., data=training, method="nnet", trControl=TControl,tuneGrid = nn_grid,linout = FALSE, trace = FALSE)
nnmodel
prediction.train <- predict(nnmodel, training[,-13], type="raw")
prediction.test <- predict(nnmodel, test[,-13], type="raw")
acctr <- confusionMatrix(prediction.train, training[,13])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,13])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="Neural Network", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

#************************
# Support Vector Machine
#************************
#linear
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 5, 10))
set.seed(12345)
svmmodel.l <- train(work_home ~., data=training, method="svmLinear", trControl=TControl, tuneGrid = grid)
svmmodel.l
prediction.train <- predict(svmmodel.l, training[,-13], type="raw")
prediction.test <- predict(svmmodel.l, test[,-13], type="raw")
acctr <- confusionMatrix(prediction.train, training[,13])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,13])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="SVM (Linear)", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

#radial
grid <- expand.grid(sigma = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1),
                    C = c(0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10, 25, 50, 75, 100))
set.seed(12345)
svmmodel.r <- train(work_home ~., data=training, method="svmRadial", trControl=TControl, tuneGrid = grid)
svmmodel.r
prediction.train <- predict(svmmodel.r, training[,-13], type="raw")
prediction.test <- predict(svmmodel.r, test[,-13], type="raw")
acctr <- confusionMatrix(prediction.train, training[,13])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,13])
accte$table
accte$overall['Accuracy']
report <- rbind(report, data.frame(Model="SVM (Radial)", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


#**************
# Final Report
#**************
# show training results
results <- resamples(list(KNN=knnmodel, C5.0=c5model, CART=cartmodel,
                          RFor=rformodel, LogReg=lrmodel, NeuNet=nnmodel,  
                          SVM.L=svmmodel.l, SVM.R=svmmodel.r))
summary(results)
dotplot(results)

# print report
report


