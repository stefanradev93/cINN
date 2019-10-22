library(abcrf)


### Set working directory to be the folder in which the script resides
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

### Load training data
X_train = read.table('./sim_data/ricker_reftable/X_train.csv', sep = ';')
theta_train = read.table('./sim_data/ricker_reftable/theta_train.csv', sep = ';')

### Load test data
X_test = read.table('./sim_data/ricker_test500/X_test.csv', sep = ';')
theta_test = read.table('./sim_data/ricker_test500/theta_test.csv', sep=';')

### 1. Fit a random forest for parameter r
train_data = data.frame(r=theta_train[,1], sum_stats=X_train)
test_data = data.frame(r=theta_test[,1], sum_stats=X_test)
rf = regAbcrf(r ~ ., data=train_data, ntree=200, ncores=8)
preds_r = predict(rf, test_data, train_data)

### 2. Fit a random forest for parameter sigma
train_data = data.frame(sigma=theta_train[,2], sum_stats=X_train)
test_data = data.frame(sigma=theta_test[,2], sum_stats=X_test)
rf = regAbcrf(sigma ~ ., data=train_data, ntree=200, ncores=8)
preds_sigma = predict(rf, test_data, train_data)


### 3. Fit a random forest for parameter rho
train_data = data.frame(rho=theta_train[,3], sum_stats=X_train)
test_data = data.frame(rho=theta_test[,3], sum_stats=X_test)
rf = regAbcrf(rho ~ ., data=train_data, ntree=200, ncores=8)
preds_rho = predict(rf, test_data, train_data)


### 4. Fit a random forest for dummy noise 
train_data = data.frame(u=theta_train[,4], sum_stats=X_train)
test_data = data.frame(u=theta_test[,4], sum_stats=X_test)
rf = regAbcrf(u ~ ., data=train_data, ntree=200, ncores=8)
preds_u = predict(rf, test_data, train_data)

theta_hat = data.frame(r=preds_r$expectation, sigma=preds_sigma$expectation,
                       rho=preds_rho$expectation, u=preds_u$expectation)

write.table(theta_hat, "./sim_data/ricker_results_rf/theta_means_rf.csv", sep=';', row.names=F)


