function [ pred ] = myregression_Kirk( trainX,testX,noutput)
% X = [ones(length(trainX),1) trainX(:,1:size(trainX,2)-noutput)]; 
%     %testX]; 
%     %trainX(:,1:size(trainX,2)-noutput);];
% X = testX;
% t = trainX(:,size(trainX,2)-noutput+1:size(trainX,2));
% w = inv(X'*X)*X'*t
% %pred = [ones(length(testX),1) testX]*w;
% pred = X*w;

% Cross-Validation
k = 3;
%section_size = floor(length(trainX)/k);
%sections = 1:section_size:length(trainX);
min = intmax('int32');
lambda =2;

for sections = 1:(floor(length(trainX)/k)):length(trainX)
%     testRows = sections(section_number) : (sections(section_number)+ section_size-1);
%     testCV = testX(testRows, :);
    CV_test_rows = sections : (sections + (floor(length(trainX))/k)-1);
    CV_test = trainX(CV_test_rows, 1:size(trainX,2)-noutput);
    CV_train_rows = setdiff((1:length(trainX)),CV_test_rows);
%    CV_train = setdiff(trainX, trainX(sections : (sections + length(trainX)/k-1), :));
    CV_train = trainX(CV_train_rows, :);
    
% Feature scaling for training data    
    X = (CV_train(:,1:size(CV_train,2)-noutput));
    
    
    
    X_mean = repmat(mean(X),length(X),1);
    X_var = repmat(sqrt(var(X)),length(X),1);
    
    CV_test_mean = repmat(mean(CV_test),length(CV_test),1);
    CV_test_var = repmat(mean(CV_test),length(CV_test),1);
    %padding for polynomial
%     X = [ones(length(X),1) X];
%     X = (X-X_mean)./repmat(sqrt(var(X)),length(X),1);
    X = (X - X_mean)./X_var;


    % linear basis function
    %phi = X;
    
    %Polynomial Basis funcion 
%     p = 1;
%     phi = X.^p;
    
    %Gaussian Basis Function
    
    phi = exp(-((X-X_mean).^2)./(2*(X_var.^2)));
    
    %Log sigmoid
%     phi = 
   


%     Feature scaling
    %X = (X-mean(X))./sqrt(var(X));
% X =[ X  ];

%     phi = [ones(length(phi),1) phi];
    t = CV_train(:,size(CV_train,2)-noutput+1:size(CV_train,2));
%     t = t - repmat(mean(t),length(t),1);

    
    
%     w = inv(phi'*phi)*phi'*t;
%    inv(A)*b is slower coz Matlab said so.....
    w = phi\t;          
   w = (inv(lambda*eye(size(phi,2)) -(phi'*phi)))*phi'*t;
   

    
%     Feature scaling
%   phi_test = (CV_test - repmat(mean(CV_test),length(CV_test),1))./sqrt(var(CV_test));
    phi_test = exp(-((CV_test-CV_test_mean).^2)./(2*(CV_test_var)));
%pad for linear
%     CV_pred = [ones(length(phi_test),1) phi_test]*w;  
    CV_pred = phi_test*w;
%     CV_pred = real(CV_pred);
    
    Y = trainX(CV_test_rows,6);
    error = norm(CV_pred - Y);
    
    if min > error
        CV_w = w;
        min = error;
    end
end   


testX = (testX - repmat(mean(testX),length(testX),1))./sqrt(var(testX));
%%pad for linear
% pred = [ones(length(testX),1) testX]*CV_w;   
pred = testX*CV_w;
pred = real(pred);

end