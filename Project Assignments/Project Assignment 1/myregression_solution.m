function [pred,w] = myregression(trainx,testx,noutputs,lambda)
[N,d] = size(trainx);
[Nt,d] = size(testx);

X = trainx(:,1:end-noutputs);
t = trainx(:,end-noutputs+1:end);

% normalization
m = mean(X);
s = std(X);
Xn = bsxfun(@rdivide,bsxfun(@minus,X,m),s);
Xntest = bsxfun(@rdivide,bsxfun(@minus,testx,m),s);

% linear regression
Phi = [Xn ones(N,1)];
PhiTest = [Xntest ones(Nt,1)];

% RBF
%M = min(100,round(.2*N));
%ind = randperm(N);
%mu = Xn(ind(1:M),:);
%s = 3;
%Phi = [exp(-distance2(Xn,mu)/(2*s^2))];
%PhiTest = [exp(-distance2(Xntest,mu)/(2*s^2))];

if(nargin<4)
    lambda = 0;
end;
w = pinv(lambda*eye(size(Phi,2))+Phi'*Phi)*Phi'*t;

pred = PhiTest*w;


