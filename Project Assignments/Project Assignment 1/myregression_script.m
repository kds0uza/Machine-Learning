% script for testing myregression.m

data = load('data_airfoil_self_noise.dat'); noutputs = 1;
[nr,nc] = size(data);

for cv = 1:100 % random cross validation
    cvindex = randperm(nr); % randomly permutes indices of data used for cv
    
    trainx = data(cvindex(1:floor(nr*4/5)),:);
    testx = data(cvindex(ceil(nr*4/5):end),1:end-noutputs);
    testt = data(cvindex(ceil(nr*4/5):end),end-noutputs+1:end);
    [pred] = myregression(trainx,testx,noutputs);
    [pred2] = myregression_solution(trainx,testx,noutputs);
    sqerr1(cv) = sum((testt(:)-pred(:)).^2);
    sqerr2(cv) = sum((testt(:)-pred2(:)).^2);
end;
mean(sqerr1)/mean(sqerr2)
% Using normalization on the data by subtracting mean and dividing by std
% dev, I get a mean squared error of around 7000 using linear regression