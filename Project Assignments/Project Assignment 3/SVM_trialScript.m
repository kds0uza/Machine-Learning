clear all;
load('SVMdata.mat');
dataset = 1;


N=400;
ind = randperm(N);
trainind=ind(1:N/2);
testind=ind(N/2+1:end);

% training data for svm
Ktrain=K1(trainind,trainind);        % training matrix 
Ytrain = Y1(trainind,1);    % class array (+1 or -1)

%testing data
Ktest=K1(trainind,testind);        % testing matrix 
Ytest = Y1(testind,1);
dataset = 1;
[prediction, alpha1, b] = SVM_trial(Ktrain, Ytrain, Ktest, dataset)

correct = sum (prediction' == Ytest);
% correct2 = sum(prediction2' == Ytest);
% stem(fTest)
% scatter(1:length(Ktest),fTest)
 function [prediction, alpha, b] = SVM_trial(Ktrain, Ytrain, Ktest, dataset)

% dataset = 1;
% if dataset < 1 || dataset > 3
%     dataset = 1;
% end
% 
% if dataset == 1
%     C = 1;
% end
% if dataset == 2
%     C = 2;
%     end
% if dataset == 3
%     C = 3;
%         end    

K = Ktrain;
Y = Ytrain;

f = -ones(size(Y,1),1)';
Aeq = Y';
% H1 = K1'.*Y1*Y1;
H = Y*Y'.*K;
lb = zeros(size(Y));
ub = dataset.*ones(size(Y));

alpha = quadprog(H, f, [], [], Aeq, 0, lb, ub);

supporters = find((alpha>(1e-3)) & (alpha < C));
% newAlpha = alpha.*supporters;     
        
% b_term3 = newAlpha.*Y';
% b_term2 = sum(b_term3*K(:,:));

% b_term2 = sum((alpha*Y')*K(:,supporters));
% b_term1 = Y(supporters)' - b_term2;
% b = mean(b_term1);

b = mean(Y(supporters)' - sum((alpha*Y')*K(:,supporters)));
% fTest_term = sum((newAlpha*Y')*Ktest);
% fTest = b + fTest_term;

% fTest_term = sum((alpha*Y')*Ktest);
% fTest = b + fTest_term;


% bb = repmat(b,size(Ktest,1),1);
% for i = 1:200
% fTest_term(i,1) = sum(newAlpha(i,1)*Y(i,1)*Ktest(i,:));    
% fTest(i,1) =  b + fTest_term(i,1);
% end

prediction = sign(sum((alpha*Y')*Ktest) + b);

% prediction = sign(fTest);
% prediction2 = sign(fTest_term);
correct = sum (prediction' == Ytest);
end

 