function [prediction, bestlambda, errors] =myWKNN(X,Y,Xtest)
% Initialization
prediction=zeros(size(Xtest,1),size(Y,2));
lambda = 0.05:0.01:1;
probC=zeros(max(Y),1);
pred=zeros(size(X,1),1);
errors=zeros(length(lambda),1);
probCT=zeros(max(Y),1);
for i=1:length(lambda)
    for j=1:size(X,1)
        x1=X(j,:);
        trainX=X(1:end ~=j,:);
        trainY=Y(1:end ~=j,:);
        R=repmat(x1,size(trainX,1),1);
        eudist=(R - trainX).^2;
        r1=sum(eudist,2);
        w=exp(-lambda(i).*r1);
        for m=1:max(Y)
            indM=(trainY==m);
            probC(m)= (w'*indM)/(sum(w));
        end
        
        [~, pred(j)]=max(probC);
    end
      
    errors(i)=sum(pred ~=Y);
end

[~, Lerr]=min(errors);
bestlambda= lambda(Lerr);

% Test data
for i=1:size(Xtest,1)
        x1t=Xtest(i,:);
        Rt=repmat(x1t,size(X,1),1);
        eudist=(Rt - X).^2;
        r1t=sum(eudist,2);
        wtest=exp(-bestlambda*r1t);
          
        for m=1:max(Y)
         indMT=(Y==m);
         probCT(m)= (wtest'*indMT)/sum(wtest);
        end
     [~, prediction(i)]=max(probCT);
end
end

