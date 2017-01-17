function [pred] = myregression(trainX, testX, noutput)
% regression parameters
lambda = [0.001, 0.01, 0.1, 0, 1, 10, 100, 1000];
% lambda = zeros(1, 10);
% for i = 1:10
%     lambda(i) = (10^(5-i));
% end
k_fold = 3;
% Gaussian Params
m = 350; %no of gaussian functions, also doubles as sigma
% sigma = [1, 10, 100, 1000, 10000];
sigma = zeros(1, 6);
for i = 1:10
    sigma(i) = 1/(m*(10^(5-i)));
end
% intializations
X_init = trainX(:, 1:(size(trainX, 2)) - noutput);
Y_init = trainX(:, (size(trainX, 2) - noutput+1):end);
% mean_Y = repmat(mean(Y_init), size(Y_init, 1), 1);
% std_Y = repmat(std(Y_init), size(Y_init, 1), 1);
% Y_norm = (Y_init-mean_Y)./std_Y;
Y_norm = Y_init;
mean_X = repmat(mean(X_init), size(X_init, 1), 1);
std_X = repmat(std(X_init), size(X_init, 1), 1);
X_norm = (X_init-mean_X)./std_X;
% Gaussian basis
% d = size(X_init, 2); %no of params going into these gaussians
mu = zeros(m, size(X_norm, 2));
for i = 1:m
    for j = 1:size(X_norm, 2)
        mu(i,j) = X_norm(randi([1, size(X_norm, 1)], 1),j);
    end
end
% mu = 2*rand(m, d) - 1;
X_basis = zeros(size(X_init, 1), m);
% for i = 1:size(X_init, 1)
%     X_basis(i, :) = (sum((exp(-((repmat(X_norm(i, :), m, 1)-mu).^2)*(m.^2/2))), 2))';
% end
sigma_min = 0;
% implement model selection (k folds)
for test = 1:length(sigma)
    for i = 1:size(X_init, 1)
       X_basis(i, :) = (exp(sum(-((repmat(X_norm(i, :), m, 1)-mu).^2)*(1/(2*(sigma(test).^2))),2)))';
    end
    w = (pinv(X_basis'*X_basis))*X_basis'*Y_norm;
    Y_est = X_basis*w;
    err = Y_norm - Y_est;
    err = 0.5*sum(err(:).^2);
%     err_debug(test, :) = [err, sigma(test)]; 
    if test == 1
        err_min = err;
        sigma_min = sigma(test);
    else if err < err_min
            err_min = err;
            sigma_min = sigma(test);
        end
    end
end

    for i = 1:size(X_init, 1)
       X_basis(i, :) = (exp(sum(-((repmat(X_norm(i, :), m, 1)-mu).^2)*(1/(2*(sigma_min.^2))),2)))';
    end
    
    
% implement regularisation parameter selection (k folds)
%K fold matrix splitting
split_length = floor(size(X_basis, 1)/k_fold);
% param_length = size(X_basis, 2);
% op_length = size(Y_norm, 2);
X_rand_index = randperm(size(X_basis, 1))';
X_basis_split = mat2cell(X_rand_index, [split_length, split_length, (size(X_rand_index, 1)-(2*split_length))], 1); 
Y_norm_split = mat2cell(X_rand_index, [split_length, split_length, (size(X_rand_index, 1)-(2*split_length))], 1); 
X_Ktrain = zeros(size(X_basis));
X_KCV = zeros(size(X_basis));
Y_Ktrain = zeros(size(Y_norm));
Y_KCV = zeros(size(Y_norm));
for test = 1:length(lambda)
    err_CV = zeros(1,size(Y_KCV,2));
    for i = 1:k_fold
        a = 1;
        for j = 1:k_fold
            if i ~= j
                X_Ktrain(((a-1)*size(X_basis_split{j}, 1)+1):((a)*size(X_basis_split{j}, 1)), :) = X_basis(X_basis_split{j}, :);
                Y_Ktrain(((a-1)*size(Y_norm_split{j}, 1)+1):((a)*size(Y_norm_split{j}, 1)), :) = Y_norm(Y_norm_split{j}, :);          
                a = a+1;
            else
                X_KCV = X_basis(X_basis_split{j}, :);
                Y_KCV = Y_norm(Y_norm_split{j}, :);
            end
        end
        X_Ktrain( ~any(X_Ktrain,2), : ) = [];
        X_KCV( ~any(X_Ktrain,2), : ) = [];
        Y_Ktrain( ~any(Y_Ktrain,2), : ) = [];
        Y_KCV( ~any(Y_KCV,2), : ) = [];
        w = (pinv(lambda(test)*eye(size(X_Ktrain'*X_Ktrain))+X_Ktrain'*X_Ktrain))*X_Ktrain'*Y_Ktrain;
        Y_est = X_KCV*w;
        err = Y_KCV - Y_est;
        err = 0.5*sum(err.^2);
        err_CV = err_CV + (err./k_fold);
    end
%     err_debug(test, :) = [err_CV, lambda(test)];
    if test == 1
        err_min = err_CV;
        lambda_min = lambda(test);
    else if err_CV < err_min
            err_min = err_CV;
            lambda_min = lambda(test);
        end
    end
end
w = (pinv(lambda_min*eye(size(X_basis'*X_basis))+X_basis'*X_basis))*X_basis'*Y_norm;
% apply basis function to test set
testX_norm = (testX - repmat(mean(X_init), size(testX, 1), 1))./repmat(std(X_init), size(testX, 1), 1);
testX_basis = zeros(size(testX_norm, 1), m);
for i = 1:size(testX, 1)
    testX_basis(i, :) = (exp(sum(-((repmat(testX_norm(i, :), m, 1)-mu).^2)*(1/(2*(sigma_min.^2))),2)))';
end
pred = testX_basis*w;
% pred = (pred_norm.*repmat(std(Y_init), size(pred_norm, 1), 1))+repmat(mean(Y_init), size(pred_norm, 1), 1); %unnormalise result
