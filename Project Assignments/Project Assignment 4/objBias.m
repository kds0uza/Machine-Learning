function bias = objBias(b,alpha,trainLabels,kernelCV)
term1 = (alpha.*trainLabels)'*kernelCV;
term2 = bsxfun(@times,-trainLabels,term1');
term3 = term2 + b;
term4 = bsxfun(@plus,1,exp(term3));
bias = -sum(bsxfun(@rdivide,1,term4));
end