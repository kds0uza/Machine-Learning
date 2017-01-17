function [prediction, bestk, errors] = myKNN(X, Y, Xtest, k)

    nTest = size(Xtest,1);
    nTrain = size(X,1);
    CV_Pred = zeros(nTrain, length(k));
    errors = zeros(1,length(k));
    
    
    
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
%             sortedCV_Train(j,:) = CV_Train(closest(j),:);
            sortedCV_Label(j,:) = CV_Label(closest(j),:);
        end
        
        % Obtaining the labels of the data at the indexes
%         CV_CloseLabels = CV_Label(closest);
        
%         for j = 1:length(k)
%             
%             CV_Pred(i,j) = mode(sortedCV_Label(1:j));
% %             pred(:,j) = CV_Pred(:,j);
%             errors(1,j) = sum(CV_Pred(:,j) ~= Ytrain);
%             [leasterror, leastErrorIndex] = min(errors);
%             goodK(i) = k(leastErrorIndex);
%         end
%             
      for j = 1:max(k)
         for m = 1:3
            indexMatch = (sortedCV_Label(1:j,:) == m);
            probClassCV(m) = sum(indexMatch)/j;
         end
         [~,CV_Pred(i,j)] = max(probClassCV);
         errors(1,j) = sum(CV_Pred(:,j) ~= Y);
         [leasterror, leastErrorIndex] = min(errors);
         goodK(i) = k(leastErrorIndex);      
      end
    
    end
    bestk = leastErrorIndex;
        

    % Testing
    
    
    
    
     testDistances = distance2(Xtest, X,2);
     [~,testClosest] = sort(testDistances,2);
     
%      for j = 1:nTest
% %             sortedCV_Train(j,:) = CV_Train(closest(j),:);
%             
%           end
     
     for i = 1:nTest
         
          
          sortedTest_Label = Y(testClosest(i,1:bestk),:);
          
            for m = 1:3
                testIndexMatch = (sortedTest_Label == m);
                probClassTest(m) = sum(testIndexMatch)/bestk;
                
            end
            
%             prediction(i,1) = mode(sortedTest_Label);
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
%     % absolute distance between all test and training data
%     dist = abs(repmat(Xtest,1,nTrain) - repmat(Xtrain(:,1)',nTest,1));




