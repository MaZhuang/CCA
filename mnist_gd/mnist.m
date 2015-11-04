%% read data
addpath mnistHelper

train_images = loadMNISTImages('train-images-idx3-ubyte');
test_images = loadMNISTImages('t10k-images-idx3-ubyte');
train_images=train_images';
test_images=test_images';

[n,p]=size(train_images);
p=p/2;

% split data into half figures
X=train_images(:,1:p);
Y=train_images(:,((p+1):(2*p)));

X_test=test_images(:,1:p);
Y_test=test_images(:,((p+1):(2*p)));

[n,px]=size(X);
py=size(Y,2);

%% parameter set up and compute the TRUE CCA directions
% k is the number of CCA dimensions, lambdax,y are CCA regularization
% parameters to avoid singularity and make sure generalization performance.
k=20;
lambdax=0.005;
lambday=0.005;

% canoncorr3 computes top k TRUE regularized CCA directions (uncentered)
% with 3 full SVDs
[A,B]=canoncorr3(X,Y,k,lambdax,lambday);
% canoncorr2 is a slight modification of matlab built in CCA to compute
% Total Correlation Captured(TCC) for TRUE CCA subspace. (the matlab built
% in CCA is centered whichle here canoncorr is uncentered. We could also
% use canoncorr3 with no regularization)
[~,~,R1]=canoncorr2(X*A,Y*B);
[~,~,R2]=canoncorr2(X_test*A,Y_test*B);

%% APPGRAD CCA
% ssx,ssy are stepsizes which need to be determined by a validation dataset
ssx=0.05;
ssy=0.05;
nrep=1;
% paramat contains the numbers of gradient iterations at which the CCA
% results are recorded and plotted
paramat=[20,100,200,500,800,1500,2500,4000];
insample=zeros(nrep,numel(paramat));
outsample=zeros(nrep,numel(paramat));

for i=1:nrep
    %gd_cca implements appgrad CCA algorithm and records the CCA subspaces
    %at after numbers of iterations specified by paramat.
    [Rx,Ry]=gd_cca(X,Y,k,lambdax,lambday,max(paramat),ssx,ssy,paramat);
    for j=1:(numel(paramat))  
       %computes Total correlations captured (TCC) for appgrad outputs
       [~,~,r1]=canoncorr2(X*Rx(:,:,j),Y*Ry(:,:,j));
       [~,~,r2]=canoncorr2(X_test*Rx(:,:,j),Y_test*Ry(:,:,j));
       %compute Proportions of Correlations Captured (PCC)
       insample(i,j)=(sum(R1)-sum(r1))/sum(R1);
       outsample(i,j)=(sum(R2)-sum(r2))/sum(R2);
    end
    fprintf('finish %d th rep \n',i)
end

%% plot results
flops=log((paramat*n*px).*(20*k));
plot(flops,100-100.*mean(insample,1),'r-+','LineWidth',1.3,'MarkerSize',10);
hold on
plot(flops,100-100.*mean(outsample,1),'b-+','LineWidth',1.3,'MarkerSize',10);


xlim([min(flops),max(flops)])
ylim([100-100*max((mean(insample,1))),100])

xlabel('log(FLOP)','FontSize',16,'FontWeight','bold')
ylabel('PCC','FontSize',16,'FontWeight','bold')
title('Mnist AppGrad','FontSize',18,'FontWeight','bold')
set(gca,'FontSize',12,'FontWeight','bold');
legend('insample','outsample')
hold off