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

set.seed(1234)
train_test_split <- sample(2, nrow(df), replace = TRUE, prob = c(0.67, 0.33))
df.train <- df[train_test_split==1, 3:31]
df.test <- df[train_test_split==2, 3:31]

df.trainLabels <- df[train_test_split==1, 32]
df.testLables <- df[train_test_split==2, 32]

position_prediction <- knn(train = df.train, test = df.test, cl = df.trainLabels, k = 10)

testLables <- data.frame(df.testLables)
merge <- data.frame(position_prediction, testLables)
names(merge) <- c("Predicted_Position", "Actual_Position")
merge$Correct[merge$Predicted_Position==merge$Actual_Position] <- 1
merge$Correct[merge$Predicted_Position!=merge$Actual_Position] <- 0

sum(merge$Correct) / length(merge$Predicted_Position)
# Model Accuracy 85%

CrossTable(x = df.testLables, y = position_prediction, prop.chisq = FALSE)
# Defense predictions 88% accurate
# Forward predictions 81% accurate
# Goalie predictions 100% accurate
# Midfield predictions 80% accurate

# See miscategorized players
merge$Name <- df[train_test_split==2,1]
View(merge[merge$Correct==0,])

# How do I know which attributes are predictive of a position?

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
# 85.7% accurate

# 