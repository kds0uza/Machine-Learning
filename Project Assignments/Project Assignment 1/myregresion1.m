
clear all;
% load data_airfoil_self_noise.dat;
% trainX = data_airfoil_self_noise;
% noutput = 1;
% testX = trainX(:,1:5);
% [pred] = myregression_Kirk2(trainX,testX,noutput);
% Y = trainX(:,6);

% 
load data_yacht_hydrodynamics.dat;
trainX = data_yacht_hydrodynamics;
noutput = 1;
testX = trainX(1:100,1:6);
[pred] = myregression_Kirk_sigmoid(trainX,testX,noutput);
Y = trainX(1:100,7);

% 
% load data_slump_test.dat;
% trainX = data_slump_test(:, 2:11);
% noutput = 3; 
% testX = trainX(:,1:7);
% [pred] = myregression_Kirk_sigmoid(trainX,testX,noutput);
% Y = trainX(:,8:10);

error_norm = norm(pred - Y);
% error = sqrt(sumsqr(pred - Y)); 
scatter(1:100, Y)
hold on
plot(1:100, pred)