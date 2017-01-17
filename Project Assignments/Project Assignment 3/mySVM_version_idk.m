function [prediction, alpha, b] = mySVM(K, Y, Kt, dataset)

if dataset >= 1 || dataset <= 3
 
    N = length(K);
    ind = randperm(N);
    trainind=ind(1:N/4);
    testind=ind((N/4)+1:end);

    % training data for svm
    CV_Train = K(trainind,trainind);        % training matrix 
    CV_Label = Y(trainind,1);               % class array (+1 or -1)

    %testing data
    CV_Test = K(trainind,testind);        % testing matrix 
    CV_TestLabel = Y(testind,1);
    correctMax = 0;
    accCVMax = 0;
    C_try = [0.1 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.5 4 4.5 5];

    for i = 1:length(C_try);
    CV_C = C_try(i);
    f = -ones(size(CV_Label,1),1)';
    Aeq = CV_Label';
    H = CV_Label*CV_Label'.*CV_Train;
    lb = zeros(size(CV_Label));
    ub = CV_C.*ones(size(CV_Label));

    CV_alpha = quadprog(H, f, [], [], Aeq, 0, lb, ub);

    CV_supporters = find((CV_alpha>(1e-3)) & (CV_alpha < CV_C));
    
    % newAlpha = alpha.*supporters;     
        
    % b_term3 = newAlpha.*CV_Label';
    % b_term2 = sum(b_term3*K(:,:));
    b = mean(CV_Label(CV_supporters)' - sum((CV_alpha*CV_Label')*CV_Train(:,CV_supporters)));

    CV_Prediction = sign(sum((CV_alpha*CV_Label')*CV_Test) + b);

    correct = sum (CV_Prediction' == CV_TestLabel);
    accCV = (correct/length(CV_TestLabel))*100;

    if accCV >= accCVMax
        accCVMax = accCV;
        correctMax = correct;
        C = CV_C;
    %     prediction = CV_prediction;
    end
end
else
C = 1;
end





% C = 1;
% f = -ones(size(Y,1),1)';
% Aeq = Y';
% H = Y*Y'.*K;
% lb = zeros(size(Y));
% ub = C.*ones(size(Y));
% 
% alpha = quadprog(H, f, [], [], Aeq, 0, lb, ub);
% 
% % supporters = ((alpha>(1e-3)) & (alpha < C));
% supporters = find((alpha>(1e-3)) & (alpha < C));
% % newAlpha = alpha.*supporters;     
% Y_sup = Y(supporters);
% 
% % b_term3 = newAlpha*Y';
% % b_term2 = sum(b_term3*K(:,:));
% % % b_term2 = sum((alpha*Y')*K(:,supporters));
% % b_term1 = Y_sup' - b_term2;
% % b = mean(b_term1);
% % 
% % b_term = (Y' - sum((alpha*Y')*K(:,:))); 
% % b = mean(b_term);
% 
% b = mean(Y(supporters)' - sum((alpha*Y')*K(:,supporters)));
% 
% % prediction = sign(sum((alpha*Y')*Kt) + b);
% prediction = sign(sum((alpha*Y')*Kt) + b);
% 





% C =1;
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

prediction = sign(sum((alpha*Y')*Kt) + b);

% prediction = sign(fTest);
% prediction2 = sign(fTest_term);
% correct = sum (prediction' == Yt);





end

 