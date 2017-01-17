 function [prediction, bestk, errors] = myKNN_T(X,Y,Xtest,k)

% Initialization for parameters
probC=zeros(max(Y),1);
prediction=zeros(size(Xtest,1),size(Y,2));
predC=zeros(size(X,1),1);
errors=zeros(length(k),1);
probCT=zeros(max(Y),1);
% To find an appropriate K value we run the following loop to obtain the
% minimum error and then set the k value accordingly to find the final
% estimates. 
for j=1:length(k)
    for i=1:size(X,1)
        % Leaving one Data point out and then going for the KNN algorithm
        x1=X(i,:);
        trainX=X(1:end ~=i,:);
        trainY=Y(1:end ~=i,:);
        R=repmat(x1,size(trainX,1),1);
        % Distance of this point with each point in the Dataset. 
        eudist=(R - trainX).^2;
        r1=sum(eudist,2);
        [~,ind]=sort(r1);
        sortY=trainY(ind(1:j));
       
        for m=1:max(trainY)
            indmatch=(sortY == m);
            probC(m)=sum(indmatch)/j;
        end
        [~,  predC(i)]=max(probC);
    end
        errors(j)=sum(predC ~=Y);
        
end
 [~, Lerr]=min(errors);
 bestk=Lerr;
 
 
 for i=1:size(Xtest,1)
     x1t=Xtest(i,:);
     Rt=repmat(x1t,size(X,1),1);
     eudist=(Rt - X).^2;
     r1t=sum(eudist,2);
     [~,indT]=sort(r1t);
     sortTY=Y(indT(1:bestk));
     for m=1:max(trainY)
         indmatchT=(sortTY ==m);
         probCT(m)= sum(indmatchT)/bestk;
     end
     [~, prediction(i)]=max(probCT);
 end
 
        
  end


