function [ pred_2 ] = myregression( trainX,testX,noutput)

trainX_1 = trainX;
trainX_2 = trainX;

testX_1 = testX;
testX_2 = testX;


% Cross-Validation
% Linear with CV and Scaling


k_1 = 5;
min_1 = 1000000000000000;
sections_1 = 1:(floor(size(trainX_1,1)/k_1)):size(trainX_1,1);
for sections_1 = sections_1(:, 1:size(sections_1,2)-1)
    CV_test_rows_1 = sections_1 : (sections_1 + (floor(size(trainX_1,1))/k_1)-1);
    CV_test_1 = trainX_1(CV_test_rows_1, 1:size(trainX_1,2)-noutput);
    CV_train_rows_1 = setdiff((1:size(trainX_1,1)),CV_test_rows_1);
    CV_train_1 = trainX_1(CV_train_rows_1, :); 
    
    %saving same matrices for use in Gaussian
    CV_test_rows_2 = CV_test_rows_1;
    CV_test_2 = CV_test_1;
    CV_train_rows_2 = CV_train_rows_1;
    CV_train_2 = CV_train_1;
    
    
    x_phi_1 = (CV_train_1(:,1:size(CV_train_1,2)-noutput));
    t_1 = CV_train_1(:,size(CV_train_1,2)-noutput+1:size(CV_train_1,2));
     
    %     Feature scaling
    x_phi_1_diff = bsxfun(@minus,x_phi_1,mean(x_phi_1));
    x_phi_1 = bsxfun(@rdivide,x_phi_1_diff,sqrt(var(x_phi_1)));
    
    CV_test_1_diff = bsxfun(@minus,CV_test_1,mean(CV_test_1));
    CV_test_1 = bsxfun(@rdivide,CV_test_1_diff,sqrt(var(CV_test_1)));
    
    x_phi_1 = [ones(length(x_phi_1),1) x_phi_1];
    

%     Linear Basis funcion 
    p = 1;
    phi_1 = x_phi_1.^p;

  for lambda_1 = 0.01:0.01:2
        
    w_1 = (pinv(lambda_1*eye(size(x_phi_1,2)) +(x_phi_1'*x_phi_1)))*(x_phi_1'*t_1);
%     w_1 = pinv(phi_1'*phi_1)*(phi_1'*t_1);
    
    CV_pred_1 = [ones(length(CV_test_1),1) CV_test_1]*w_1;
    

    Y_1 = trainX_1(CV_test_rows_1, size(CV_train_1,2) - noutput+1:size(CV_train_1,2));

    error_1 = sum((CV_pred_1 - Y_1).^2);
    
    if min_1 > error_1
        CV_w_1 = w_1;
        min_1 = error_1;
    end
end
   

testX_1 = bsxfun(@rdivide, bsxfun(@minus, testX_1, mean(testX_1)),sqrt(var(testX_1)));
pred_1 = [ones(length(testX_1),1) testX_1]*CV_w_1;





% Gaussian Basis Function
[nr,nc] = size(trainX_2);


min_2 = 1000000000;

    
    x_phi_2 = (CV_train_2(:,1:size(CV_train_2,2)-noutput));
    t_2 = CV_train_2(:,size(CV_train_2,2)-noutput+1:size(CV_train_2,2));
    [nr,nc] = size(CV_train_2);

    %with Scaling
%     x_phi = (x_phi - mean(x_phi))./sqrt(var(x_phi));
%     CV_test_2 = (CV_test_2 - mean(x_phi))./sqrt(var(x_phi));

   
    pred_min_2 = [];
    
    m = randperm(nr);
    
    mew_2 = CV_train_2(m,1:size(CV_train_2,2)-noutput); %,size(x_phi,2));
    sig_2 = 1/10;    

    
%Calculating phi and phi_test    
M_phi_2 = zeros(size(mew_2));
M_phi_test_2 = zeros(size(mew_2));
big_phi_2 =[];
big_phi_test_2 = [];
    
for i = 1:length(CV_train_2)
    M_phi_2 = repmat(x_phi_2(i,:), size(mew_2,1),1);  

    
    phi_2(i, :) = sum(exp((-bsxfun(@rdivide,((bsxfun(@minus,M_phi_2,mew_2)).^2),(2*sig_2^2)))),2);
    
    big_phi_2 = [big_phi_2 ; phi_2(i,:) ];
end    

for i = 1:length(CV_test_2)

    M_phi_test_2 = repmat(CV_test_2(i,:), size(mew_2,1),1);

    
      phi_test_2(i, :) = sum(exp((-bsxfun(@rdivide,((bsxfun(@minus,M_phi_test_2,mew_2)).^2),(2*sig_2^2)))),2);
    big_phi_test_2 = [big_phi_test_2 ; phi_test_2(i, :)];
end

% lambda_2 = 0.01;
% w_2 = pinv(lambda_2*eye(size(big_phi_2))+(big_phi_2'*big_phi_2))*(big_phi_2'*t_2);
w_2 = pinv(big_phi_2'*big_phi_2)*(big_phi_2'*t_2);

CV_pred_2 = big_phi_test_2*w_2;


 Y_2 = trainX_2(CV_test_rows_2, size(CV_train_2,2) - noutput+1:size(CV_train_2,2));
    error_2 = sum((CV_pred_2 - Y_2).^2);
    
    if min_2 > error_2
        CV_w_2 = w_2;
        min_2 = error_2;
    end
end   



big_phi_testX_2 = [];
for i = 1:length(testX_2)

    M_phi_testX_2 = repmat(testX_2(i,:), size(mew_2,1),1);

    
    phi_testX_2(i, :) = sum(exp((-bsxfun(@rdivide,((bsxfun(@minus,M_phi_testX_2,mew_2)).^2),(2*sig_2^2)))),2);
     
    big_phi_testX_2 = [big_phi_testX_2 ; phi_testX_2(i, :)];
end
pred_2  = big_phi_testX_2*CV_w_2;



    %Model Selection

if (min_1 < min_2)

    pred = [ones(length(testX_1),1) testX_1]*CV_w_1;
else

    pred = big_phi_testX_2*CV_w_2;
end


end













