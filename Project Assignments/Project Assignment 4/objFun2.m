function fun2 = objFun2(alpha,trainLabels,kernel,C)
bigG1 = -bsxfun(@times,alpha,log(alpha));
bigOne = ones(size(alpha,1),1);
bigG2 = bsxfun(@times,(bigOne - alpha),log(bigOne - alpha));
bigG3 = bigG1 + bigG2;
bigG = sum(bigG3);
bigAlpha = -(1/(2)*C)*((alpha.*trainLabels)'*kernel*(alpha.*trainLabels));
fun2 = bigAlpha + bigG;
end