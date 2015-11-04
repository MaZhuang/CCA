function[Ox,Oy]=sgd_cca2(X,Y,k,lambdax,lambday,bs,outputs,ssx,ssy)
%stochastic appgrad CCA algorithm, k is the target CCA dimension, lambadas
%are regularization parameters, bs is the minibatch size for computing
%gradients in every iteration, outputs are the numbers of iterations at
%which the CCA subspaces will be stored, ssx,ssy are stepsizes.

[n,px]=size(X);
py=size(Y,2);
% sp is the subsapmle size needed for the normalization step (see paper for
% detail), the sample size depend on k, size of the matrix to be normalized
% empirically we find 10k is good for our datasets.
sp=10*k;

X=X';
Y=Y';

%create a three way array that stores the CCA subspaces
m=numel(outputs);
Ox=zeros(px,k,m);
Oy=zeros(py,k,m);
current=1;


% random initialize a subspace in Y
Wy=randn(py,k);

% initialize the auxilary varibles at 0 (the varibles with tilde in the paper )
wxt=zeros(px,k);
wyt=zeros(py,k);

niter1=max(outputs);

% stochastic gradient iterations
for s=1:niter1
    ssx1=ssx(current);
    ssy1=ssy(current);
    
    %sgd updates the auxilary varible of x
    id=randsample(n,bs);
    Xt=X(:,id);
    Yt=Y(:,id);
    gwxt=(Xt*(Xt'*wxt)-Xt*(Yt'*Wy))./bs+wxt.*lambdax;
    wxt=wxt-ssx1.*gwxt;
    
    
    % update the non tilde version of x
    id1=randsample(n,sp);
    TP=(X(:,id1))'*wxt;
    TP1=sqrtm(TP'*TP./sp+lambdax.*eye(k));
    Wx=wxt/TP1;
    
    % same update for y
    id=randsample(n,bs);
    Xt=X(:,id);
    Yt=Y(:,id);
    gwyt=(Yt*(Yt'*wyt)-Yt*(Xt'*Wx))./bs+wyt.*lambday;
    wyt=wyt-ssy1.*gwyt;
    
    TP=(Y(:,id1))'*wyt;
    TP1=sqrtm(TP'*TP./sp+lambday.*eye(k));
    Wy=wyt/TP1;
    
    if mod(s,1000)==0
        disp(['finish', num2str(s),'iteration'])
    end
    
    % store the subspace at required iterations
     if s==outputs(current)
        Ox(:,:,current)=Wx;
        Oy(:,:,current)=Wy;
        current=current+1;
        
    end
end


   
    


