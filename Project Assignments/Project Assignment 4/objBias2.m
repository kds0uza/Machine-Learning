function bias2 = objBias2(b,alpha,trainLabels,kernelCV)
% term1 = alpha.*trainLabels;
% term2 = kernelCV*term1;
% term3 = term2 + b;
% term4 = -trainLabels.*term3;
% term5 = log(1 + exp(term4));
bias2 = sum(log(1 + exp(-trainLabels.*(kernelCV*((alpha.*trainLabels)/C) + b))));
end