function [A,B,r,U,V,stats] = canoncorr2(X,Y)
%CANONCORR Canonical correlation analysis.
%   [A,B] = CANONCORR(X,Y) computes the sample canonical coefficients for
%   the N-by-P1 and N-by-P2 data matrices X and Y.  X and Y must have the
%   same number of observations (rows) but can have different numbers of
%   variables (cols).  A and B are P1-by-D and P2-by-D matrices, where D =
%   min(rank(X),rank(Y)).  The jth columns of A and B contain the canonical
%   coefficients, i.e. the linear combination of variables making up the
%   jth canonical variable for X and Y, respectively.  Columns of A and B
%   are scaled to make COV(U) and COV(V) (see below) the identity matrix.
%   If X or Y are less than full rank, CANONCORR gives a warning and
%   returns zeros in the rows of A or B corresponding to dependent columns
%   of X or Y.
%
%   [A,B,R] = CANONCORR(X,Y) returns the 1-by-D vector R containing the
%   sample canonical correlations.  The jth element of R is the correlation
%   between the jth columns of U and V (see below).
%
%   [A,B,R,U,V] = CANONCORR(X,Y) returns the canonical variables, also
%   known as scores, in the N-by-D matrices U and V.  U and V are computed
%   as
%
%      U = (X - repmat(mean(X),N,1))*A and
%      V = (Y - repmat(mean(Y),N,1))*B.
%
%   [A,B,R,U,V,STATS] = CANONCORR(X,Y) returns a structure containing
%   information relating to the sequence of D null hypotheses H0_K, that
%   the (K+1)st through Dth correlations are all zero, for K = 0:(D-1).
%   STATS contains eight fields, each a 1-by-D vector with elements
%   corresponding to values of K:
%
%      Wilks:    Wilks' lambda (likelihood ratio) statistic
%      chisq:    Bartlett's approximate chi-squared statistic for H0_K,
%                with Lawley's modification
%      pChisq:   the right-tail significance level for CHISQ
%      F:        Rao's approximate F statistic for H0_K
%      pF:       the right-tail significance level for F
%      df1:      the degrees of freedom for the chi-squared statistic,
%                also the numerator degrees of freedom for the F statistic
%      df2:      the denominator degrees of freedom for the F statistic
%
%   Example:
%
%      load carbig;
%      X = [Displacement Horsepower Weight Acceleration MPG];
%      nans = sum(isnan(X),2) > 0;
%      [A B r U V] = canoncorr(X(~nans,1:3), X(~nans,4:5));
%
%      plot(U(:,1),V(:,1),'.');
%      xlabel('0.0025*Disp + 0.020*HP - 0.000025*Wgt');
%      ylabel('-0.17*Accel + -0.092*MPG')
%
%   See also PRINCOMP, MANOVA1.

%   References:
%     [1] Krzanowski, W.J., Principles of Multivariate Analysis,
%         Oxford University Press, Oxford, 1988.
%     [2] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.

%   Copyright 1993-2009 The MathWorks, Inc.
%   $Revision: 1.1.8.1 $  $Date: 2010/03/16 00:12:35 $

if nargin < 2
    error('stats:canoncorr:TooFewInputs','Requires two arguments.');
end

[n,p1] = size(X);
if size(Y,1) ~= n
    error('stats:canoncorr:InputSizeMismatch',...
          'X and Y must have the same number of rows.');
elseif n == 1
    error('stats:canoncorr:NotEnoughData',...
          'X and Y must have more than one row.');
end
p2 = size(Y,2);


% Factor the inputs, and find a full rank set of columns if necessary
[Q1,T11,perm1] = qr(X,0);
rankX = sum(abs(diag(T11)) > eps(abs(T11(1)))*max(n,p1));
if rankX == 0
    error('stats:canoncorr:BadData',...
          'X must contain at least one non-constant column');
elseif rankX < p1
    warning('stats:canoncorr:NotFullRank','X is not full rank.');
    Q1 = Q1(:,1:rankX); T11 = T11(1:rankX,1:rankX);
end
[Q2,T22,perm2] = qr(Y,0);
rankY = sum(abs(diag(T22)) > eps(abs(T22(1)))*max(n,p2));
if rankY == 0
    error('stats:canoncorr:BadData',...
          'Y must contain at least one non-constant column');
elseif rankY < p2
    warning('stats:canoncorr:NotFullRank','Y is not full rank.');
    Q2 = Q2(:,1:rankY); T22 = T22(1:rankY,1:rankY);
end

% Compute canonical coefficients and canonical correlations.  For rankX >
% rankY, the economy-size version ignores the extra columns in L and rows
% in D. For rankX < rankY, need to ignore extra columns in M and D
% explicitly. Normalize A and B to give U and V unit variance.
d = min(rankX,rankY);
[L,D,M] = svd(Q1' * Q2,0);
A = T11 \ L(:,1:d) * sqrt(n-1);
B = T22 \ M(:,1:d) * sqrt(n-1);
r = min(max(diag(D(:,1:d))', 0), 1); % remove roundoff errs

% Put coefficients back to their full size and their correct order
A(perm1,:) = [A; zeros(p1-rankX,d)];
B(perm2,:) = [B; zeros(p2-rankY,d)];

% Compute the canonical variates
if nargout > 3
    U = X * A;
    V = Y * B;
end

