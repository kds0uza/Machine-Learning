function [prediction, alpha, b] = mySVM(K, Y, Kt, dataset)
if nargin > 3
if find([1,2,3]==dataset)
    load('SVMdata.mat');
    if dataset == 1
        Ktrain = K1;
        Ytrain = Y1;
    else if dataset == 2
            Ktrain = K2;
            Ytrain = Y2;
        else
            Ktrain = K3;
            Ytrain = Y3;
        end
    end
    C_trial = [0.001, 0.01 ,0.1, 1, 10, 100, 1000];
    for i = 1:length(C_trial)
        C_test = C_trial(i);
        err_total = 0;
        for j = 1:10
            %Training and CV matrix sizes
            train_size = floor(0.8*(size(Ktrain, 1)));
            tot_size = size(Ktrain, 1);
            %shuffle training and test matrices
            rand_arr = randperm(tot_size);
            Ktrain = Ktrain(rand_arr, rand_arr);
            Ytrain = Ytrain(rand_arr);
            %SVM Training and CV error computation
            H = (Ytrain(1:train_size)*Ytrain(1:train_size)').*Ktrain(1:train_size, 1:train_size);
            Aeq = Ytrain(1:train_size)';
            Beq = 0;
            f = -1*ones(train_size,1);
            lb = zeros(train_size, 1);
            ub = C_test*ones(train_size, 1);
            alpha = quadprog(H, f, [], [], Aeq, Beq, lb, ub);

            % Making alpha C and 0
            % alpha(alpha >(C_test*1e-03)) = C;
            % alpha(alpha <(C_test*1e-03)) = 0;

            b = Ytrain(1:train_size) - sum(alpha*Ytrain(1:train_size)'*Ktrain(1:train_size, 1:train_size),1)';
            b = mean(b);
            fxtest = sum((alpha*Ytrain(1:train_size)'*Ktrain(1:train_size, (train_size+1):tot_size)),1)' + b;
            errcount = sum(sign(fxtest) ~= Ytrain((train_size+1):tot_size))/size(fxtest, 1);
            err_total = err_total+errcount;
        end
        if i == 1
            err_min = err_total;
            C_min = C_test;
        else if err_total < err_min
                err_min = err_total;
                C_min = C_test;
            end
        end
    end
    C = C_min;
else
    C = 1;
end
else
    C = 1;
end


%train kernel with cross validated C, K and Y matrices
H = (Y*Y').*K;
Aeq = Y';
Beq = 0;
f = -1*ones(size(K, 1),1);
lb = zeros(size(K, 1), 1);
ub = C*ones(size(K, 1), 1);
alpha = quadprog(H, f, [], [], Aeq, Beq, lb, ub);
% Making alpha C and 0
alpha(alpha >(C*1e-03)) = C;
alpha(alpha <(C*1e-03)) = 0;
b = Y - sum((alpha*Y'*K),1)';
b = mean(b);
%test kernel predictions
fxtest = sum((alpha*Y'*Kt),1)' + b;
prediction = sign(fxtest);
end
    

