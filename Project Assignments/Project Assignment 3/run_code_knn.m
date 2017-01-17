% clear all;
% load ('KNNdata.mat');
load ('SVMdata.mat');
% dataset = [1 2 3];
% [prediction, bestk, errors ] = myKNN_dev(Xtrain, Ytrain, Xtest, [1:959]);
% [prediction, bestk, errors ] = myKNN(Xtrain, Ytrain, Xtest, [1:959]);
% [prediction, bestlambda ] = myWKNN(Xtrain, Ytrain, Xtest);
% [prediction, bestlambda ] = myWKNN_dev(Xtrain, Ytrain, Xtest, [0.001:0.001:0.01]');
% [prediction, bestlambda ] = myWKNN_Aswin(Xtrain, Ytrain, Xtest);
% Kt4 = K1(201:400,201:400);
% Yt4 = Y1(201:400);
% Kt1 = K1(1:200,1:200);
% Yt1 = Y1(1:200);
% Kt2 = K1(101:200,201:400);
% Yt3 = Y1(101:200);
% Kt3 = K1(201:400,1:200);
% Yt4 = Y1(201:300);
% Kt = [Kt1 Kt2; Kt3 Kt4];
% Yt = [Yt1; Yt2; Yt3; Yt4];
% Kt = [K1(201:400,:); K1(1:200,:)];
% Yt = [Y1(201:400); Y1(1:200)];



N=length(K1);
ind = randperm(N);
testRowInd=ind(1:300);
testColInd=ind(1:100);
Kt = K1(testRowInd,testColInd);
Yt = Y1(testColInd,1);
% Kt = Kt(:, 200:end);
% Yt = Yt(200:end);
dataset = 5;
[prediction, alpha, b] = mySVM_Kirk2(K1(1:300,1:300), Y1(1:300), Kt, dataset);
correct = sum (prediction' == Yt);
accuracy = (sum (prediction' == Yt)/length(Yt))*100;



% SVMStruct = svmtrain(Kt,Yt);
% fitSVM = fitcsvm(K1.*K1',Y1);
% SVMacc = sum(prediction == SVMStruct.GroupNames);