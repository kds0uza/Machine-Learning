function [ pred ] = myregression_Kirk_final( trainX,testX,noutput)

% X = [ones(length(trainX),1) trainX(:,1:size(trainX,2)-noutput)]; 
%     %testX]; 
%     %trainX(:,1:size(trainX,2)-noutput);];
% X = testX;
% t = trainX(:,size(trainX,2)-noutput+1:size(trainX,2));
% w = inv(X'*X)*X'*t
% %pred = [ones(length(testX),1) testX]*w;
% pred = X*w;

trainX_1 = trainX;
trainX_2 = trainX;
trainX_3 = trainX;
trainX_4 = trainX;
testX_1 = testX;
testX_2 = testX;
testX_3 = testX;
testX_4 = testX;


% Linear with CV and Scaling
% 
%     trainX = (trainX-mean(trainX))./sqrt(var(trainX));
%     testX = (testX - mean(trainX))./sqrt(var(tX));

% Cross-Validation

k = 15;
min = 1000000000000000;
sections = 1:(floor(size(trainX,1)/k)):size(trainX,1);
for sections = sections(:, 1:size(sections,2)-1)%1:(floor(size(trainX,1)/4)):size(trainX)
    CV_test_rows = sections : (sections + (floor(size(trainX_1,1))/k)-1);
    CV_test = trainX_1(CV_test_rows, 1:size(trainX_1,2)-noutput);
    CV_train_rows = setdiff((1:size(trainX_1,1)),CV_test_rows);
    CV_train = trainX_1(CV_train_rows, :); 
    
    x_phi = (CV_train(:,1:size(CV_train,2)-noutput));
    t = CV_train(:,size(CV_train,2)-noutput+1:size(CV_train,2));
     
    %     Feature scaling
%    x_phi = (x_phi-mean(x_phi))./sqrt(var(x_phi));
    x_phi = bsxfun(@rdivide, bsxfun(@minus, x_phi, mean(x_phi)), std(x_phi));
    CV_test = bsxfun(@rdivide, bsxfun(@minus, CV_test, mean(x_phi)), std(x_phi));
    %CV_test = (CV_test - mean(x_phi))./sqrt(var(x_phi));
    
%     lambda_I = [zeros(size(x_phi,1),1) eye(size(x_phi))];
%     lambda_I = [zeros(size(lambda_I,2),1) ; lambda_I];
    
    x_phi = [ones(length(x_phi),1) x_phi];
    

%     Polynomial Basis funcion 
    p = 1;
    phi = x_phi.^p;

   
    lambda = 2.61;
    w = phi\t;           
%     w = (inv(lambda*eye(size(x_phi,2)) +(x_phi'*x_phi)))*(x_phi'*t);
    
    
    CV_pred = [ones(length(CV_test),1) CV_test]*w;
    
    Y = trainX_1(CV_test_rows,6);
%     error = norm(CV_pred - Y);
    error = sum((CV_pred - Y).^2);
    
    if min > error
        CV_w = w;
        min = error;
    end
end   

testX_1 = (testX_1 - mean(testX_1))./sqrt(var(testX_1));
pred = [ones(length(testX_1),1) testX_1]*CV_w;
% Y = trainX(:,6);
% error_norm = norm(pred - Y);




%     %logarithmic regression 
%     for i = 1:size(trainX_2,1)
%         for j = 1:size(trainX_2,2)
%             
%             trainX_2(i,j) = 1/(1-exp(trainX_2(i,j)));
%         end
%     end
%     
%     for i = 1:size(testX_2,1)
%         for j = 1:size(testX_2,2)
%             
%             testX_2(i,j) = 1/(1-exp(testX_2(i,j)));
%         end
%     end
% %     







end













