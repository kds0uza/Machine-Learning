clear;
clc;
load('SVMdata.mat');
Ksplit = K2;
Ysplit = Y2;
randinit = randperm(size(Ksplit, 1));
Ksplit = Ksplit(randinit, randinit);
Ysplit = Ysplit(randinit);
%Training and CV matrix sizes
train_size = floor(0.8*(size(Ksplit, 1)));

K = Ksplit(1:train_size, 1:train_size);
Y = Ysplit(1:train_size);
Kt = Ksplit(1:train_size, (train_size+1):end);
Yt = Ysplit((train_size+1):end);


[prediction, alpha, b] = mySVM_Aswin(K2, Y2, Ksplit, 2);

err_per = (sum(prediction ~= Ysplit)/size(prediction, 1))*100; %Error in percentage