function [S] = AssetPathsCorrelated(S0,r,y,sig,corr,t_mat,nruns,antithetic)
% Function to generate correlated sample paths for assets assuming
% geometric Brownian motion.
%
% S = AssetPathsCorrelated(S0,mu,sig,corr,dt,steps,nsims,antithetic)
%
% Inputs: S0 - stock price
%       : r - riskless rate
%       : y - dividend yields
%       : sig - volatility
%       : corr - correlation matrix
%       : t_mat - time to expiry
%       : dt - size of time steps
%       : nruns - number of simulation paths to generate
%       : antithetic - whether to use antithetic variates or not.
%           1 = Yes, 0 = No
%
% Output: S - a 3-dimensional matrix where each row represents a time step, 
%             each column represents a seperate simulation run and each 
%             3rd dimension represents a different asset.
%

% edited by Laks May,2014

% get the number of assets
nAssets = length(S0);

% adjust the sigmas to keep all stock prices as 100
sig = (sig./S0)* 100 ;
% now assign all stock prices to 100
S0 = [100 100 100];
%expected return accounting for the continuous dividend yield
r = r - y;
%calculate steps as years to maturity * 365
steps = t_mat * 365;
%size of time steps:
dt = 1/365;
% calculate the drift
nu = r - sig.*sig/2;

% do a Cholesky factorization on the correlation matrix
R = chol(corr);

% generate correlated random sequences and paths
% If using antithetic variates:
if antithetic == 1
    % pre-allocate the output
    S = nan(steps+1,nruns*2,nAssets);
    for idx = 1:nruns
    % generate uncorrelated random sequence
    x = randn(steps,size(corr,2));    
    % correlate the sequences
    ep = x*R;
    % Generate potential paths
    S(:,idx,:) = [ones(1,nAssets); exp(cumsum(repmat(nu*dt,steps,1)+...
        ep*diag(sig)*sqrt(dt)))]*diag(S0);
    S(:,nruns+idx,:) = [ones(1,nAssets);...
        exp(cumsum(repmat(nu*dt,steps,1)- ep*diag(sig)*sqrt(dt)))]*diag(S0);
    end
% If not using antithetic variates
elseif antithetic == 0
    % Pre-allocate the output
    S = nan(steps+1,nruns,nAssets);
    for idx = 1:nruns
    % generate uncorrelated random sequence
    x = randn(steps,size(corr,2));   
    % correlate the sequences
    ep = x*R;
    % Generate potential paths
    S(:,idx,:) = [ones(1,nAssets);exp(cumsum(repmat(nu*dt,steps,1)+...
        ep*diag(sig)*sqrt(dt)))]*diag(S0);
    end
end

% If only one simulation then remove the unitary dimension
if nruns==1
    S = squeeze(S);
    S2 = squeeze(S2);
end 

% extract the points of exercise of the barrier option