function[A,B]= canoncorr3(X,Y,k,lambdax,lambday)

[n,px]=size(X);
py=size(Y,2);

wx=sqrtm(X'*X+(n*lambdax).*eye(px));
wy=sqrtm(Y'*Y+(n*lambday).*eye(py));

[u,~,v]=svds((wx\(X'*Y))/wy,k);

A=wx\u;
B=wy\v;

