clear;clc;
load('KNNdata.mat')
[predWKNN, bestlambda] = myWKNN(Xtrain, Ytrain, Xtest);
[predKNN, bestK, errorsKNN] = myKNN(Xtrain, Ytrain, Xtest, 1:(length(Ytrain)-1));
mod = fitcknn(Xtrain, Ytrain);
modpred = predict(mod, Xtest);
errWKNN = sum(modpred ~= predWKNN);
errKNN = sum(modpred ~= predKNN);