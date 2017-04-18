function delta_tau=tax_depr(x,Y,b,r);


Y_star = Y*(1-(1/b)) ;
beta = b/Y ;

%dist = (((x/(x+r))*exp(-(x+r)))-(((beta/(beta+r))*(1-exp(-(beta+r)*Y_star)))...
%    +(((exp(-beta*Y_star))/(r*(Y-Y_star)))*(exp(-r*Y_star)-exp(-r*Y)))))^2 ;

%dist = ((x*(1/(1-((1-x)/(1+r)))))-(((beta/(beta+r))*(1-exp(-(beta+r)*Y_star)))...
%    +(((exp(-beta*Y_star))/(r*(Y-Y_star)))*(exp(-r*Y_star)-exp(-r*Y)))))^2 ;

dist = ((x/(x+r))-(((beta/(beta+r))*(1-exp(-(beta+r)*Y_star)))...
    +(((exp(-beta*Y_star))/(r*(Y-Y_star)))*(exp(-r*Y_star)-exp(-r*Y)))))^2 ;


delta_tau= dist;
