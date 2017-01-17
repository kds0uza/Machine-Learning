function [pred] = myregression_sam(trainX, testX, noutput)

trainX1 = trainX;
trainX2 = trainX;
testX1 = testX;
testX2 = testX;

% model 1
% Scaling to geet the values in the range [0,1]
for j = 1:size(trainX1,2)-noutput
    for i = 1:length(trainX1)
        trainX1(i,j) = (trainX1(i,j) - min(trainX1(:,j)))/(max(trainX1(:,j))-min(trainX1(:,j)));
    end
end

for j = 1:size(testX1,2)
    for i = 1:length(testX1)
        testX1(i,j) = (testX1(i,j) - min(testX1(:,j)))/(max(testX1(:,j))-min(testX1(:,j)));
    end
end

mu_train = mean(trainX1(:,1:size(trainX1,2)-noutput));
sigma_train = var(trainX1(:,1:size(trainX1,2)-noutput));

mu_train = repmat(mu_train,length(trainX1),1);
sigma_train = repmat(sigma_train,length(trainX1),1);

mu_test = mean(testX1(:,1:size(testX1,2)));
sigma_test = var(testX1(:,1:size(testX1,2)));

mu_test = repmat(mu_test,length(testX1),1);
sigma_test = repmat(sigma_test,length(testX1),1);

phi_i1 = exp(-(trainX1(:,1:size(trainX1,2)-noutput) - mu_train).^2./(2*sigma_train.^2));
phi_o1 = exp(-(testX1(:,1:size(testX1,2)) - mu_test).^2./(2*sigma_test.^2));

w1 = (inv(phi_i1'*phi_i1))*(phi_i1'*trainX1(:,size(trainX1,2)-noutput+1:size(trainX1,2)));
y1 = phi_i1*w1;
out1 = trainX1(:,size(trainX1,2)-noutput+1:size(trainX1,2));
err1 = norm(y1-out1);

% model 2
% Using the Gaussian basis function

mu_train = mean(trainX2(:,1:size(trainX2,2)-noutput));
sigma_train = var(trainX2(:,1:size(trainX2,2)-noutput));

mu_train = repmat(mu_train,length(trainX2),1);
sigma_train = repmat(sigma_train,length(trainX2),1);

mu_test = mean(testX2(:,1:size(testX2,2)));
sigma_test = var(testX2(:,1:size(testX2,2)));

mu_test = repmat(mu_test,length(testX2),1);
sigma_test = repmat(sigma_test,length(testX2),1);

phi_i2 = exp(-(trainX2(:,1:size(trainX2,2)-noutput) - mu_train).^2./(2*sigma_train.^2));
phi_o2 = exp(-(testX2(:,1:size(testX2,2)) - mu_test).^2./(2*sigma_test.^2));

w2 = (inv(phi_i2'*phi_i2))*(phi_i2'*trainX2(:,size(trainX2,2)-noutput+1:size(trainX2,2)));
y2 = phi_i2*w2;
out2 = trainX2(:,size(trainX2,2)-noutput+1:size(trainX2,2));
err2 = norm(y2-out2);

% model selection
if err1<err2
    pred = phi_o1*w1;
else
    pred = phi_o2*w2;
end
