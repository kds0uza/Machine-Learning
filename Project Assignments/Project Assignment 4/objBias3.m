function bias3 = objBias3(b,alpha,trainLabels,kernelCV,C)
supporters = find(alpha > 1e-5);
bias3 = sum(log(1 + exp(-trainLabels(supporters,:)'.*(((alpha.*trainLabels)/C)'*kernelCV(:,supporters) + b))));
end