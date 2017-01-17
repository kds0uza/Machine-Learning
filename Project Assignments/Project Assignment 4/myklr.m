%% Question 2

% Kernelized Logistic Regressison
close all
clear

% Loading the Data
x = load('heartstatlog_trainSet.txt');
y = load('heartstatlog_trainLabels.txt');
%std=max(x)-min(x);
x = bsxfun(@rdivide,bsxfun(@minus,x,mean(x)),std(x));
        
xtest=load('heartstatlog_testSet.txt');
ytest=load('heartstatlog_testLabels.txt');
%std1=max(xtest)-min(xtest);
xtest = bsxfun(@rdivide,bsxfun(@minus,xtest,mean(xtest)),std(xtest));
   
% Setting class2= -1 so that our activation function works
y(y==2)=-1;
ytest(ytest==2)=-1;
n=length(x);
lambda=[0.01 0.05 0.25 1 5 25 100];
E1=zeros(length(lambda),1);
E2=zeros(length(lambda),1);
E3=zeros(length(lambda),1);

%% Finding Optimal \lambda

% Running for varying lambda.
for i=1:length(lambda)
        cvr=randperm(size(x,1));
        xtrain= x(cvr(1:floor((4/5)*n)),:);
        ytrain=y(cvr(1:floor((4/5)*n)));
        xcv=x(cvr(floor((4/5)*n)+1:n),:);
        ycv=y(cvr(floor((4/5)*n)+1:n));
        
        
        k=basis(xtrain,xtrain,1);
        
        [alphatrain] = DlogReg(ytrain,lambda(i),k);
        
        b(i)=biasval(ytrain,alphatrain,k,lambda(i));
        %b=0;
        [ypredtrain(i,:),E1(i)]= error(ytrain,alphatrain,ytrain,k,b(i),lambda(i));
        
        kcv= basis(xtrain,xcv,1);
        [ypredcv(i,:),E2(i)] =error(ytrain,alphatrain,ycv,kcv,b(i),lambda(i));
        
        ktest= basis(xtrain,xtest,1);
        [ypredtest(i,:), E3(i)]=error(ytrain,alphatrain,ytest,ktest,b(i),lambda(i));
    
    
end
%% Plotting

% Plotting for error
[~,ind]=min(E2);
finalpred=ypredtest(ind,:)';
err= E3(ind);
figure(1)
plot(log10(lambda),E1,'-o')
hold on
plot(log10(lambda),E2,'-o')
plot(log10(lambda),E3,'-o')
xlabel('log_{10}{\lambda}')
ylabel('Error')
legend('Training Error','CV Error', 'Testing Error')
title('Error values for varying \lambda')

disp(['Optimal value of lambda is ', num2str(lambda(ind))])
disp((['Optimal value of b is ', num2str(b(ind))]))

%% Kernelization

% Function to call various basis functions
function phix = basis( xi, xj, n)

switch n
    case 1
        % Polynomial basis function;
        d=1;
        phix = (xi*xj').^d;
          
    case 2
        % RBF kernel
        sig=1;
        for i=1:size(xi,1)
            for j=1:size(xj,1)      
                phix(i,j)= exp(sig*(norm((xi(i,:)-xj(j,:)),2)^2));
            end
        end
        
        
           
end
   
end
%% Calculating /alpha

% Function to get the values of alpha 
function   alpha = DlogReg( y,lambda,K )
N = length(y);
alpha = ones(N,1);
options = optimset('Display','off');
%C=1/lambda;
C=lambda;
alpha = fmincon(@(alpha) func(alpha,y,K,C), alpha, [],[],y',0,zeros(N,1),C*ones(N,1),[],options);
end
%% Lagrange function for /alpha

% Largrangian function for alpha
function Lagalpha  = func(alpha,y,K,l)
g = @(delta) delta.*log(delta) + (1-delta).*log(1-delta); 
Lagalpha = 0.5*(alpha*alpha').*(y*y').*K + l * sum(g(alpha/l));
Lagalpha = sum(Lagalpha(:));
end

%% Error and Prediction

% Function to calculate error and the predictions
function [predict,E]= error(y,alpha,ycv,k,b,l)
n=size(ycv,1);
%b=ycv-(sum((alpha*y')*kt))';
%be=mean(b);
%ypred=sum((alpha.*y')*k)'+b;
ypred=(((alpha.*y)/l)'*k)'+b;
predict=sign(ypred);
E= sum(predict~=ycv)/n;
end
%% Optimal value of Bias
% Function to get an optimal value for b
 function b= biasval(y,alpha,K,l)
 %w=sum((alpha*y')*kt)';
 class1=find(alpha>1e-5); %& alpha<lambda-(1e-5));
 %fun= @(b) sum((-y'.*exp(-y'.*((alpha.*y)'*K)))./(1+exp(-y'.*((alpha.*y)'*K+b))));
 funcb=@(b) sum(log(1+exp(-y(class1)'.*(((alpha.*y)/l)'*K(:,class1)+b))));
 %funcb=@(b) sum(log(1+exp(-y'.*(((alpha.*y)'*K)+b))));
 b=fminunc(funcb,0);
 
 end
% 
