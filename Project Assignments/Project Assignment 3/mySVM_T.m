function [prediction, alpha, b] =mySVM_T(K,Y,Kt,dataset)

 if dataset == [1,2,3]
     C=1:15;
     n=length(K);
     Ktrain= K(1:floor((4/5)*n),1:floor((4/5)*n));
     Ytrain= Y(1:floor((4/5)*n));
     Kt1=K(1:floor((4/5)*n),floor((4/5)*n)+1:n);
     Yt1=Y(floor((4/5)*n)+1:n);
     E=zeros(length(C),1);
 
     for i=1:length(C)
        [predict, ~, ~]=quadcall(Ktrain,Ytrain,Kt1, C(i));
        E(i)= sum(predict~=Yt1);
     end
     [~, indC]= min(E);
     %Err= sum(prediction1~=Yt1);
     %perE= (Err/size(Kt1,2))*100;
     bestC= C(indC);
    % min(E)
 else 
   bestC=1;
   %perE=0;
 end
   % Test data
    [prediction, alpha, b]=quadcall(K,Y,Kt, bestC);
    
end 

    function [prediction, alpha,be]=quadcall(K,Y,Kt, C)
        H=Y*Y'*K;
        H=(H+H')./2;
        f=-ones(1, size(K,1) ); %,1);
        A=[];
        b=[];
        Aeq=Y';
        beq=0;
        lb=zeros(size(K,1),1);
        ub=C*ones(size(K,1),1);
        options = optimset('MaxIter',1000,'LargeScale','off');
        alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub ,[],options);
        class1=find(alpha>1e-3); % & alpha<(C-1e-3));
        %numclass=length(class1);
        %b=(Y(class1)-(sum((alpha(class1)*Y(class1)')*K(class1)))');
        b=(Y(class1)-(sum((alpha*Y')*K(:,class1)))');
        %size(b)
        %plot(b)
        be=mean(b);
        %size(be)
        ftest=sum((alpha*Y')*Kt)'+ be;
        prediction=sign(ftest);
    end 
   
