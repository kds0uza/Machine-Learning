function [prediction, CV_alpha, b] = mySVM(K, Y, Kt, dataset)

if dataset >= 1 && dataset <= 3
    Ctry = [1 1.25 1.5 1.75 2 2.5 3 3.5 4 5];
    goodC = zeros(5,1);
    for i = 1:5
        N = length(K);
        ind = randperm(N);
        trainind=ind(1:floor(N*0.8));
        testind=ind(floor(N*0.8)+1:end);

        % training data for svm
        CV_Train = K(trainind,trainind);        % training matrix 
        CV_Label = Y(trainind,1);               % class array (+1 or -1)

        %testing data
        CV_Test = K(trainind,testind);        % testing matrix 
        CV_TestLabel = Y(testind,1);

        accMaxCV = 0;
    

        for j = 1:length(Ctry)
            CV_C = Ctry(j);
            f = -ones(size(CV_Label,1),1)';
            Aeq = CV_Label';
            H = CV_Label*CV_Label'.*CV_Train;
            lb = zeros(size(CV_Label));
            ub = CV_C.*ones(size(CV_Label));

            CV_alpha = quadprog(H, f, [], [], Aeq, 0, lb, ub);
            CV_supporters = find(CV_alpha > 1e-3 & CV_alpha < CV_C);

            b_mat = CV_Label(CV_supporters)' - sum((CV_alpha*CV_Label')*CV_Train(:,CV_supporters));
            CV_b = mean(b_mat);

            pred = sign(CV_b + sum(CV_alpha*CV_Label')*CV_Test)';
        
            accCV = (sum(pred == CV_TestLabel)/length(CV_TestLabel))*100;
        
            if accCV > accMaxCV
                accMaxCV = accCV;
                goodC(i) = Ctry(j);
            end
        end
    
    end
C = mode(goodC);
end
if dataset ~= [1,2,3]
    C = 1;
end
    
%     
%     [predCV, alphaCV, bCV] = SVM_magic(CV_train, CV_Label, CV_test)
% CV_Accuracy(CV_C,1) = sum(pred == CV_TestLabel)/length(CV_TestLabel);
% end
% 
% [~,bestCInd] = min(CV_Accuracy);
% C = Ctry(bestCInd);
% else
%     C = 1;
%     [prediction, alpha, b] = SVM_magic(K, Y, Kt)
% end




f = -ones(size(Y,1),1)';
Aeq = Y';
H = Y*Y'.*K;
lb = zeros(size(Y));
ub = C.*ones(size(Y));

alpha = quadprog(H, f, [], [], Aeq, 0, lb, ub);
supporters = find(alpha > 1e-3 & alpha < C);

b_mat = Y(supporters)' - sum((alpha*Y')*K(:,supporters));
b = mean(b_mat);

prediction = sign(b + sum(alpha*Y')*Kt);


end


 