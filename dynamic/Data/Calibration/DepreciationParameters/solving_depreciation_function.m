%%% This program solves for the geometric rate of depreciation that gives
%%% a present value of depreciation deductions that is the same as
%%% the present value of depreciation for given method of tax deprecation
%%% and asset life.

%% Note the IRS assigns each different asset a depreciation method.  These
%% include straight-line (SL), declining balance (DB), and the lowest of 
%% DB and SL (DBSL). The IRS also assigns the length of time over which
%% an asset will be depreciated.


%-------------------------------------------------------------------------%
% Parameters                                                              %
%-------------------------------------------------------------------------%
% Y   = Asset depreciable life (in years)
% b   = rate of acceleration (e.g. if double declining balance is used,
%       b=2)
% r   = real interest rate
%-------------------------------------------------------------------------%
Y = 5 ;
b = 2 ;
r = 0.05 ;

options_solve=optimset('TolX',1e-18,'TolFun',1e-18,'Display','off');

%% call to non-linear solver %%
delta_tau=fsolve(@(x) tax_depr(x,Y,b,r), 0, options_solve)


%% solve by hand %%
Y_star = Y*(1-(1/b)) ;
beta = b/Y ;
c = (((beta/(beta+r))*(1-exp(-(beta+r)*Y_star)))...
    +(((exp(-beta*Y_star))/(r*(Y-Y_star)))*(exp(-r*Y_star)-exp(-r*Y)))) ;
delta_tau2 = r/((1/c)-1)


r= 0.07 ;
%% call to non-linear solver %%
delta_tau=fsolve(@(x) tax_depr(x,Y,b,r), 0, options_solve)


%% solve by hand %%
Y_star = Y*(1-(1/b)) ;
beta = b/Y ;
c = (((beta/(beta+r))*(1-exp(-(beta+r)*Y_star)))...
    +(((exp(-beta*Y_star))/(r*(Y-Y_star)))*(exp(-r*Y_star)-exp(-r*Y)))) ;
delta_tau2 = r/((1/c)-1)


r= 0.1 ;
%% call to non-linear solver %%
delta_tau=fsolve(@(x) tax_depr(x,Y,b,r), 0, options_solve)


%% solve by hand %%
Y_star = Y*(1-(1/b)) ;
beta = b/Y ;
c = (((beta/(beta+r))*(1-exp(-(beta+r)*Y_star)))...
    +(((exp(-beta*Y_star))/(r*(Y-Y_star)))*(exp(-r*Y_star)-exp(-r*Y)))) ;
delta_tau2 = r/((1/c)-1)


