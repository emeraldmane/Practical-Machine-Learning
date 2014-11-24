testdata <- read.csv(file = 'pml-testing.csv', na.strings = c('NA','#DIV/0!',''))
traindata <- read.csv(file = 'pml-training.csv', na.strings = c('NA','#DIV/0!',''))

dim(traindata); dim(testdata)

## Preprocessing Training Data

blankdata <- which(traindata=="",arr.ind=T)
traindata[blankdata]<-NA

traindata_na = sapply(traindata, function(x) {sum(is.na(x))})
table(traindata_na)

traindata_na <- traindata[ , colSums(is.na(traindata)) == 0]
dim(traindata_na)

## Preprocessing Test Data

blankdata <- which(testdata=="",arr.ind=T)
testdata[blankdata]<-NA

testdata_na = sapply(testdata, function(x) {sum(is.na(x))})
table(testdata_na)

testdata_na <- testdata[ , colSums(is.na(testdata)) == 0]
dim(testdata_na)

## Removing unnecessary non numerical values

traindata_new <-traindata_na[,-c(1:7)]
testdata_new <-testdata_na[,-c(1:7)]

dim(traindata_new); dim(testdata_new)

library(caret)

zeroVar = nearZeroVar(traindata_new[sapply(traindata_new, is.numeric)], saveMetrics = TRUE)
traindata_nonzerovar = traindata_new[,zeroVar[, 'nzv']==0]
dim(traindata_nonzerovar)

correlmatrix <- cor(na.omit(traindata_nonzerovar[sapply(traindata_nonzerovar, is.numeric)]))
dim(correlmatrix)

## Removing Variables with high correlation

no_correl_var = findCorrelation(correlmatrix, cutoff = .90, verbose = T)
traindata_nocorrel = traindata_nonzerovar[,-no_correl_var]
dim(traindata_nocorrel)

## Subsetting Training data into 60% train and 40% test subsets

set.seed(1200)
subsets <- createDataPartition(y=traindata_nocorrel$classe, p=0.6, list=F)
traindata_subset <- traindata_nocorrel[subsets, ] 
testdata_subset <- traindata_nocorrel[-subsets, ]
dim(traindata_subset); dim(testdata_subset)

## Random Forest Model

RF_model <- train(classe ~ .,
                  data = traindata_subset, 
                  method = 'rf',
                  prox = T,
                  trControl = trainControl(method = "cv", 
                                          number = 4, 
                                          allowParallel = T, 
                                          verboseIter = T))
print(RF_model)


RF_model_predict <- predict(RF_model,testdata_subset)

RF_model_confusionmatrix <- confusionMatrix(RF_model_predict,testdata_subset$classe)

RF_model_confusionmatrix

### Out of Sample Error

dim(testdata_subset)

out_sample_errorpredict <- predict(RF_model, testdata_subset)

out_sample_error_accuracy <- sum(out_sample_errorpredict == testdata_subset$classe)/length(out_sample_errorpredict)

out_sample_error <- 1 - out_sample_error_accuracy
out_sample_error

err <- out_sample_error*100
paste0("Out of Sample Error is ", round(err,digits=2),"%")

### TEST Data testing

testing_prediction <- predict(RF_model, testdata_new)
testing_prediction <- as.character(testing_prediction)
testing_prediction

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(testing_prediction)


