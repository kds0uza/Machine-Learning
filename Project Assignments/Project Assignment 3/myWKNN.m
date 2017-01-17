function [prediction, bestLambda] = myWKNN(X, Y, Xtest)

    nTest = size(Xtest,1);
    nTrain = size(X,1);
    
%     CV_Pred = zeros(nTrain, length(k));
%     errors = zeros(1,length(k));
    lambdaValues = [0.02:0.0001:0.03];
    nLambda = size(lambdaValues,2);
    
 for lambda = 1:nLambda
    for i = 1:nTrain
        
        CV_Test = X(i,:);
%         CV_Train(i,:) = setdiff(Xtrain, CV_Test, 'stable');
       
        % deleting the ith row(validation)from the training data and labels
        CV_Train=X(1:end ~= i,:);
        CV_Label=Y(1:end ~= i,:);
               
        % Calculating distances from testing data point to each training
        % data point
        distances = distance2(CV_Train, CV_Test,2); 
        
        % Getting indexes of the data points after sorting the distances in
        % ascending order
        [~,closest] = sort(distances);
                
        for j = 1:nTrain - 1
            sortedCV_Label(j,:) = CV_Label(closest(j),:);
            w(j) = exp(-lambdaValues(lambda)*distances(closest(j),1));
        end
        
        for m = 1:3
            indexMatch = (sortedCV_Label == m);
            probClassCV(m) = sum(w'.*indexMatch)/sum(w);
        end
         
        [~, pred(i,1)] = max(probClassCV);
    end
        errors(lambda) = sum(pred ~= Y);
 end
 
 [leastError, leastErrorIndex] = min(errors);
  
 bestLambda = lambdaValues(leastErrorIndex);

    % Testing
   
     testDistances = distance2(Xtest, X,2);
     [~,testClosest] = sort(testDistances,2);
         
     for i = 1:nTest         
          
          sortedTest_Label = Y(testClosest(i,1:29),:);
          wTest = exp(-bestLambda*testDistances(i, testClosest(i, 1:29)));
            for m = 1:3
                testIndexMatch = (sortedTest_Label == m);
                probClassTest(m) = sum(wTest'.* testIndexMatch)/sum(wTest);
            end
            [~,prediction(i,1)] = max(probClassTest);
        
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
