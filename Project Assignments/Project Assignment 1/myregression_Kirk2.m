function [ pred ] = myregression_Kirk2( trainX,testX,noutput)
    
trainX_2 = trainX;
testX_2 = testX;

% Cross-Validation

k = 3;
min = 1000000000000000;
sections = 1:(floor(size(trainX,1)/k)):size(trainX,1);

for sections = sections(:, 1:size(sections,2)-1)%1:(floor(size(trainX,1)/4)):size(trainX)
    CV_test_rows_2 = sections : (sections + (floor(size(trainX_2,1))/k)-1);
    CV_test_2 = trainX_2(CV_test_rows_2, 1:size(trainX_2,2)-noutput);
    CV_train_rows_2 = setdiff((1:size(trainX_2,1)),CV_test_rows_2);
    CV_train_2 = trainX_2(CV_train_rows_2, :); 


    
    x_phi = (CV_train_2(:,1:size(CV_train_2,2)-noutput));
    t = CV_train_2(:,size(CV_train_2,2)-noutput+1:size(CV_train_2,2));
    

%     x_phi = (trainX(:,1:size(trainX,2)-noutput));
%     t = trainX(:,size(trainX,2)-noutput+1:size(trainX,2));
%     act_op = trainX(:,size(trainX,2)
    
    %with Scaling
%     x_phi = (x_phi - mean(x_phi))./sqrt(var(x_phi));
%     testX = (testX - mean(x_phi))./sqrt(var(x_phi));
    min = 100000000000;
    pred_min = [];
    
   
    mew = rand(1000,size(x_phi,2));
    sig = 1/1000;    

M_phi = zeros(size(mew));
M_phi_test = zeros(size(mew));
big_phi =[];
big_phi_test = [];
    
for i = 1:length(CV_train_2)
    M_phi = repmat(x_phi(i,:), size(mew,1),1);  
    phi(i,:) = sum(exp((-(M_phi - mew).^2)./(2*sig)),2);
    big_phi = [big_phi ; phi(i,:) ];
end    

for i = 1:length(CV_test_2)

    M_phi_test = repmat(CV_test_2(i,:), size(mew,1),1);
    phi_test(i, :) = sum(exp((-(M_phi_test - mew).^2)./(2*sig)),2);
    big_phi_test = [big_phi_test ; phi_test(i, :)];
end


%Sigmoidal

% for i = 1:length(trainX)
%     M_phi = repmat(x_phi(i,:), size(mew,1),1);  
%     phi(i,:) = sum(1/(1+exp((M_phi - mew))./(sqrt(sig))),2);
%     big_phi = [big_phi ; phi(i,:) ];
% end    
% 
% for i = 1:length(testX)
% 
%     M_phi_test = repmat(testX(i,:), size(mew,1),1);
%     phi_test(i,:) = sum(1/(1+exp((M_phi_test - mew))./(sqrt(sig))),2);
%     big_phi_test = [big_phi_test ; phi_test(i, :)];
% end

% w = big_phi\t;
w = pinv(big_phi'*big_phi)*(big_phi'*t);

CV_pred = big_phi_test*w;



Y = trainX_2(CV_test_rows_2,6);
%     error = norm(CV_pred - Y);
    error = sum((CV_pred - Y).^2);
    
    if min > error
        CV_w = w;
        min = error;
    end
end   

testX_2 = (testX_2 - mean(testX_2))./sqrt(var(testX_2));
pred  = testX_2*CV_w;
% pred = [ones(length(testX_2),1) testX_2]*CV_w;






% error  = norm(pred1-trainX(1:100,7));

% if min > error
%         pred_min = pred1;
%         min = error;










% for sections = 1:length(M):length(trainX)    
%     M_test_rows = sections : (sections + length(M)-1);
%     M_test = trainX(M_test_rows, 1:size(trainX,2)-noutput);
%     M_train = trainX(M_test_rows, 1:size(trainX,2));
% %     CV_train_rows = setdiff((1:length(trainX)),M_test_rows);
% %     CV_train = trainX(CV_train_rows, :); 
%     t = M_train(:,size(M_train,2)-noutput+1:size(M_train,2));
% 
% 
%     mew = rand(100,size(M_test,2));
%     sig = 1/mew;
%     
%     phi = exp((-(M_test - mew).^2)./(2*sig));
%     
%     phi_test = exp((-(M_test - mew).^2)./(2*sig));
%     
%     w =phi\t;
%     
%     pred = phi_test*w;
    
end   



