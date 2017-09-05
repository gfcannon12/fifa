library(class)
library(gmodels)

fifa_dataset <- read.csv('./fifa.csv')
head(fifa_dataset)
str(fifa_dataset)
summary(fifa_dataset)

table(fifa_dataset$National_Position)

# Filter to select only players that are starters on national teams.
# We are getting the Name, National Position, and all of the 1-99 attribute columns
df <- fifa_dataset[fifa_dataset$National_Position != '' & fifa_dataset$National_Position != 'Sub', c(1, 3, 20:48)]

df$Position_Group[df$National_Position=='GK'] <- 'Goalie'
df$Position_Group[df$National_Position %in% c('CB', 'LB', 'RB', 'RWB', 'LWB', 'RCB', 'LCB')] <- 'Defense'
df$Position_Group[df$National_Position %in% c('CM', 'LCM', 'RCM', 'CAM', 'CDM', 'RAM', 'LAM', 'RDM', 'LDM', 'RM', 'LM')] <- 'Midfield'
df$Position_Group[df$National_Position %in% c('RW', 'LW', 'RS', 'LS', 'ST', 'CF', 'RF', 'LF') ] <- 'Forward'

# To check for NaNs
# sapply(df, function(x) sum(is.nan(x)))

set.seed(7) # You can vary this for different splits and will get different results
train_test_split <- sample(2, nrow(df), replace = TRUE, prob = c(0.67, 0.33))
df.train <- df[train_test_split==1, 3:31]
df.test <- df[train_test_split==2, 3:31]

df.trainLabels <- df[train_test_split==1, 32]
df.testLables <- df[train_test_split==2, 32]

position_prediction <- knn(train = df.train, test = df.test, cl = df.trainLabels, k = 9)

testLables <- data.frame(df.testLables)
merge <- data.frame(position_prediction, testLables)
names(merge) <- c("Predicted_Position", "Actual_Position")
merge$Correct[merge$Predicted_Position==merge$Actual_Position] <- 1
merge$Correct[merge$Predicted_Position!=merge$Actual_Position] <- 0

sum(merge$Correct) / length(merge$Predicted_Position)
# Model Accuracy 86% with seed 7 

CrossTable(x = df.testLables, y = position_prediction, prop.chisq = FALSE)
# Predictions with seed 7
# Defense 93% accurate
# Forward 86% accurate
# Goalie 100% accurate
# Midfield 76% accurate

# See miscategorized players
merge$Name <- df[train_test_split==2,1]

# Linear Discrimination Analysis
trainFactor <- as.factor(df.trainLabels)
testFactor <- as.factor(df.testLables)
library(MASS)
lda_model <- lda(x=df.train, grouping=trainFactor)
lda_predict <- predict(lda_model, newdata=df.test)

# Assess the accuracy of the prediction
# percent correct for each category of G
ct <- table(testFactor, lda_predict$class)
diag(prop.table(ct, 1))
# total percent correct
sum(diag(prop.table(ct)))
# 89% accurate with seed 7 (my manual tests have ranged from 80-89)

# Multinomial Logistic Regression https://stats.idre.ucla.edu/r/dae/multinomial-logistic-regression/
library(nnet)
df.trainLabels2 <- as.factor(df.trainLabels)
df.trainLabels2 <- relevel(df.trainLabels2, ref='Goalie')
reg <- multinom(df.trainLabels2 ~ ., data=df.train)
summary(reg)
z <- summary(reg)$coefficients/summary(reg)$standard.errors
p <- (1 - pnorm(abs(z), 0, 1)) * 2
p
# These p-values all suggest that none of the values are significant

# Refining KNN lesson https://www3.nd.edu/~steve/computing_with_data/17_Refining_kNN/refining_knn.html

# The lm formula likes the labels to be included in the dataframe
df.train2 <- df.train
df.train2$Position <- df.trainLabels

# Example of linear regression of how Position determines Ball Control
lm1 <- lm(Ball_Control ~ Position, data = df.train2)
summary(lm1)$fstatistic[1]

# Creating a vector to store f statistic for each var
exp_var_fstat <- as.numeric(rep(NA, times = 29))
names(exp_var_fstat) <- names(df.train)

exp_vars <- names(df.train)

# Doing the linear regression for each var and storing f stats
for (j in 1:length(exp_vars)) {
  exp_var_fstat[exp_vars[j]] <- summary(lm(as.formula(paste(exp_vars[j], " ~ Position")), 
                                           data = df.train2))$fstatistic[1]
}

sort(exp_var_fstat, decreasing = TRUE)

# Creating a dataframe for each feature
df_train_L <- lapply(exp_vars, function(x) {
  df <- data.frame(sample = rownames(df.train), variable = x, value = df.train[, x], class = df.trainLabels)
  df
})

head(df_train_L[[1]])

names(df_train_L) <- exp_vars

# Here is how to create the f stats vector using plyr (cleaner, more flexible)
library(plyr)
var_sig_fstats <- laply(df_train_L, function(df) {
  fit <- lm(value ~ class, data = df)
  f <- summary(fit)$fstatistic[1]
  f
})

names(var_sig_fstats) <- names(df_train_L)

most_sig_stats <- sort(var_sig_fstats, decreasing = T)
df_train_ord <- df.train[, names(most_sig_stats)]

# Monte Carlo cross-validation
df_train_ord$id <- seq(from=1, to=343)
rownames(df_train_ord) <- df_train_ord$id
names(df.trainLabels) <- df_train_ord$id

# Further dividing the training set
length(df_train_ord$Ball_Control) # 343 - size of the data set
(2/3) * length(df_train_ord$Ball_Control) # 229 size of the training data set
length(df_train_ord$Ball_Control) - 229 # 114 size of the test data set

# Creating 100 different combinations of sampling for training and validation
training_family_L <- lapply(1:100, function(j) {
  perm <- sample(1:343, size = 343, replace = F)
  shuffle <- df_train_ord$id[perm]
  trn <- shuffle[1:234]
  trn
})

validation_family_L <- lapply(training_family_L, function(x) setdiff(df_train_ord$id, x))

# Trying to find the optimal number of vars and k value

# Varying the number of vars
N <- seq(from = 3, to = 29, by = 2)

# Rule of thumb is that we should take the sqrt of the size of the reference set for k
sqrt(length(training_family_L[[1]])) # 15.2

# Varying the number of K
K <- seq(from = 3, to = 19, by = 2)

# Demonstrating what the KNN looks like at n=5, k=7 for the first train/validation split
knn_test <- knn(train = df_train_ord[training_family_L[[1]], 1:5], test = df_train_ord[validation_family_L[[1]], 
                        1:5], cl = df.trainLabels[training_family_L[[1]]], k = 7)

knn_cross <- CrossTable(x = df.trainLabels[validation_family_L[[1]]], y = knn_test, prop.chisq = FALSE)
error <- 1 - sum(diag(knn_cross$prop.tbl)) # 75.2% accurate
error # 27.5% error (72.5% accuracy)


# j = index, n = length of range of variables, k=k
core_knn <- function(j, n, k) {
  knn_predict <- knn(train = df_train_ord[training_family_L[[j]], 1:n], 
                     test = df_train_ord[validation_family_L[[j]], 1:n], cl = df.trainLabels[training_family_L[[j]]], 
                     k = k)
  knn_cross <- CrossTable(x = df.trainLabels[validation_family_L[[j]]], y = knn_predict, prop.chisq = FALSE)
  error <- 1 - sum(diag(knn_cross$prop.tbl))
  error
}

# You can use the core_knn function to test the error rate for various combinations
core_knn(1, 5, 7) # 27.5% error (matches the above)

# Creating an empty data frame to store 100 tests of each K and N combination (12,600 rows)
param_df1 <- merge(data.frame(mc_index = 1:100), data.frame(var_num = N))
param_df <- merge(param_df1, data.frame(k = K))
str(param_df)

# This is a test using plyr to fill a df with only the first 20 test data sets
knn_err_est_df_test <- ddply(param_df[1:20, ], .(mc_index, var_num, k), function(df) {
  err <- core_knn(df$mc_index[1], df$var_num[1], df$k[1])
  err
})
head(knn_err_est_df_test)

# This will run through all 12,600 rows using plyr (Every test, every k, every n)
str_time <- Sys.time()
knn_err_est_df <- ddply(param_df, .(mc_index, var_num, k), function(df) {
  err <- core_knn(df$mc_index[1], df$var_num[1], df$k[1])
  err
})
time_lapsed <- Sys.time() - str_time
save(knn_err_est_df, time_lapsed, file = "knn_err_est_df.RData") # This took 2.16 min to run!

# Renaming the error column as error
names(knn_err_est_df)[4] <- "error"

# Make a subset df for all the tests with a certain n and k
mean_ex_df <- subset(knn_err_est_df, var_num == 5 & k == 7)
mean(mean_ex_df$error)

# Do this for all n and all k
mean_errs_df <- ddply(knn_err_est_df, .(var_num, k), function(df) mean(df$error))
names(mean_errs_df)[3] <- "mean_error"
mean_errs_df[which.min(mean_errs_df$mean_error), ]
# Optimal number of variables (n) is 21, optimal neighbors (k) is 7
# For the 100 tests with this combination, mean error is 15.1% (lowest among all combinations of n and k)

# Validation on the true test set
df_test_ord <- df.test[, names(df_train_ord[1:29])]
df_test_pred <- knn(train = df_train_ord[, 1:21], df_test_ord[, 1:21], df.train2$Position, 
                   k = 7)

test_table <- table(df_test_pred, df.testLables)
test_table

sum(diag(test_table))/sum(test_table)
# 85% correct on test data set
