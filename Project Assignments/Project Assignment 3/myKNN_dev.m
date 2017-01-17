function [ prediction, bestk, errors ] = myKNN_dev( X, Y, Xtest, k )

numTrain = size(X, 1);
numK = length(k);
predictedClassCV = zeros(numTrain, numK);

    for i = 1:numTrain
        testCV = X(i, :);
        
        %All elements except testCV element 
        trainCV = X;
        YtrainCV = Y;
        trainCV(i, :) = [];
        YtrainCV(i) = [];
        
        distances = distance2(trainCV, testCV, 2);
        
        [~, nearestIndices] = sort(distances);
        
        for j = 1:numK
            nearestClassValues = YtrainCV(nearestIndices(1:k(j)), 1);
            predictedClassCV(i, j) = mode(nearestClassValues);
        end
    end
    
    errors = zeros(numK,1);
    for i = 1:numK
       pred = predictedClassCV(:, i);
       errors(i,1) = sum(pred ~= Y);
    end
        
    [~, leastError] = min(errors);
    
    bestk = k(leastError);
    
    %Testing
    numTest = size(Xtest,1);
    testDistances = distance2(Xtest, X, 2);
    
    [~, sortedTestIndices] = sort(testDistances,2);
    
    prediction = zeros(numTest, 1);
    for i = 1:numTest
        nearestTestClassValues = Y(sortedTestIndices(i, 1:bestk), 1);
        prediction(i) = mode(nearestTestClassValues);
    end
end

function [D]=distance2(X,Y,A)
% computes sq. A-norm distance between two D-dimensional vectors (rows of X
% and Y)
% X - [Nx x D] matrix
% Y - [Ny x D] matrix
% A - [D x D] matrix (optional input)
if(nargin<3)
    A=eye(size(X,2));
end;

    D=bsxfun(@plus,bsxfun(@plus,-2*X*A*Y',sum(X*A.*X,2)),[sum(Y*A.*Y,2)]');
end