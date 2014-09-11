function [ price, sd ] = BasketPricing( S0,K,r,y,sig,corr,t_mat,nexercise,nruns,antithetic )
% Function to price Bermudan Call on a basket of assets
%
% [ price sd] = BasketPricing(
% S0,K,r,y,sig,corr,t_mat,nexercise,nruns,antithetic)
%
% Inputs: S0 - stock price
%       : K - strike price
%       : r - riskless rate (annualized, constinously compounded)
%       : y - dividend yields
%       : sig - volatility
%       : corr - correlation matrix
%       : t_mat - time to expiry
%       : nexercise - number of exercise opportunities
%       : nruns - number of simulation paths to generate
%       : antithetic - whether to use antithetic variates or not.
%           1 = Yes, 0 = No
%
% Output: price - Option value
%         sd - sample standard deviation of the estimator
%
% Size of time step
dt = 1/365;
% Generate price paths
P = AssetPathsCorrelated(S0,r,y,sig,corr,t_mat,nruns,antithetic);
% Sort the prices of the assets at each node
P = sort(P,3);
% Define the highest price
X1 = P(:,:,3);
% Define the second highest price
X2 = P(:,:,2);
% Size of the gap between the exercise opportunities
gap = round((t_mat*365+1)/nexercise); 
% Discount Factor
DF = exp(-r*dt*gap);
% Terminal value
CF = max(P((t_mat*365)+1,:,3)- K,0);

for i = ((t_mat*365+1)-gap):-gap:2
    % Discounted CF from t = i + 1
    DCF = CF*DF;
    % Payoff if exercise
    CF = max(P(i,:,3)-K,0);
    % Index of in-the-money paths
    index = CF > 0;
    % Run regressions
    % Define independent variables
    X = [ones(sum(index),1) X1(i,index)' X2(i,index)' X1(i,index)'.^2 ...
        X2(i,index)'.^2 (X1(i,index)'.*X2(i,index)')];
    % Regression coefficients
    beta = X\DCF(index)';
    % Value of hold strategy
    hold = max([ones(size(index))' X1(i,:)' X2(i,:)' X1(i,:)'.^2 ...
        X2(i,:)'.^2 (X1(i,:)'.*X2(i,:)')] * beta,0);
    % Compare exercise vs hold strategies
    index = CF <= hold';
    % Set CF = discounted cash flow if not exercising
    CF(index) = DCF(index);
end
if antithetic == 1
    % Average the payoff from sample path and anti-sample path
    payoff = exp(-r*dt*mod(t_mat*365+1,nexercise))*(CF(1:nruns) + CF(nruns+1:2*nruns))/2;
    % Modulus is used to account for the residual gap after the last exercise opp in backward induction
elseif antithetic == 0
    payoff = exp(-r*dt*mod(t_mat*365+1,nexercise))*CF;
end
% Results
price = mean(payoff);
sd = std(payoff)/sqrt(nruns);
end

