function [Ox,Oy]=gd_cca(X,Y,k,lambdax,lambday,niter1,ssx,ssy,outputs)
% appgrad cca function, k is target CCA subspace dimension, lambdas are
% regularization parameters,outputs are the numbers of iterations at which
% the CCA subspace are stored. ssx ssy are stepsizes, niter1 is the laregst
% number of iterations.

[n,px]=size(X);
py=size(Y,2);

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

% gradient iterations
for s=1:niter1
    %gradient updates the auxilary varible of x
    gwxt=(X'*(X*wxt)-X'*(Y*Wy))./n+wxt.*lambdax;
    wxt=wxt-ssx.*gwxt;
    
    % update the non tilde version of x
    TP=X*wxt;
    TP1=sqrtm(TP'*TP+(n*lambdax).*eye(k));
    Wx=wxt/TP1;
    Wx=Wx.*sqrt(n);
    
    % same update for y
    gwyt=(Y'*(Y*wyt)-Y'*(X*Wx))./n+wyt.*lambday;
    wyt=wyt-ssy.*gwyt;
    
    TP=Y*wyt;
    TP1=sqrtm(TP'*TP+(n*lambday).*eye(k));
    Wy=wyt/TP1;
    Wy=Wy.*sqrt(n);
    
    % store the subspace at required iterations
    if s==outputs(current)
        Ox(:,:,current)=Wx;
        Oy(:,:,current)=Wy;
        current=current+1;
    end
    
    if mod(s,100)==0
        disp(['finish ', num2str(s), 'th iteration'])
    end
end


