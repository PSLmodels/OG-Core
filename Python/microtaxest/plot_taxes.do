/* plot_taxees.do*/

*------------------------------------------------------------------*
/* This script reads in data generated from the OSPC Tax Calcultor and
the 2009 IRS PUF.  It then plots these data to help in visualizing
the relationshiop between average effective tax rates and marginal
effective rates rates and income and age. */
*------------------------------------------------------------------*;




#delimit ;
set more off;
capture clear all;
capture log close;
set memory 8000m;
cd "~/repos/dynamic/Python/microtaxest/" ;
log using "~/repos/dynamic/Python/microtaxest/plot_taxes.log", replace ;


local datapath "/Users/jasondebacker/repos/microsimint/Data" ;
local graphpath "~/repos/dynamic/Python/microtaxest/Graphs" ;

set matsize 800 ;



/* Read in data from Tax Calculator */
insheet using "`datapath'/2015_tau_n.csv", comma clear ;

/* rename variables */
rename v1 id ;
rename mtrwage mtr_wage ;
rename mtrselfemployedwage mtr_labor ;
rename wageandsalaries wages ;
rename selfemployedincome se_inc ;
rename wageselfemployedincome labor_inc ;
rename adjustedtotalincome ati ;
rename totaltaxliability tot_tax ;

/* create some variables */
replace labor_inc = wages + se_inc ;
gen aetr = tot_tax/ati ;
gen capital_inc = ati - labor_inc ;
gen ln_ati = log(ati) ;
gen has_se_inc = se_inc != 4 ;

/* create income bins */
scalar bin_size = 10000 ;
scalar max_ati = 2000000 ;
scalar num_bins = ceil(max_ati/bin_size) ;
gen ati_bin = 0 if ati <= 0 ;
replace ati_bin = 1 if ati > 0 & ati <= bin_size ;
local i = 1 ;
while `i'< num_bins { ; /* not sure why, but wasn't working with scalar in the "to" place */
	replace ati_bin = (`i'+1) if (ati > bin_size*`i') & (ati <= bin_size*(`i'+1)) ;
	local i = `i'+1 ;
} ;

/* create age categories */
gen age_bin = 1 if age < 25 ;
replace age_bin = 2 if age >= 25 & age < 34 ;
replace age_bin = 3 if age >=35 & age < 44 ;
replace age_bin = 4 if age >=45 & age < 54 ;
replace age_bin = 5 if age >=55 & age < 64 ;
replace age_bin = 6 if age >=65 & age < 74 ;
replace age_bin = 7 if age >=75  ;



/* summarize the data before any deletions */
summ [iweight=weight] ;

/* Set min allowable AETR values for estimation as functions of the tax parameters
 Note that these bounds are important as you can get some outliers in these 
 ratios, especially for filer with ATI near zero */
scalar max_eitc_rate = 0.45 ; 
scalar max_statutory_mtr = 0.396 ;
scalar min_statutory_mtr = 0.1 ;

scalar max_allowable_aetr = max_statutory_mtr*1.3 ;
scalar min_allowable_aetr = -0.5*(max_eitc_rate-min_statutory_mtr) ;


/* summarize data after some restrictions on ATI and AETRs */
summarize if aetr < max_allowable_aetr & aetr > min_allowable_aetr & ati > 5 [iweight=weight] ;

/* plot curves by ati bin and age */
collapse (mean) ati aetr tot_tax, by(ati_bin age_bin) ;
gen aetr_bin = tot_tax/ati ;
twoway (connected aetr_bin ati_bin if age_bin == 1 & aetr_bin < 0.5, msize(small) mcolor(blue) lcolor(blue) msymbol(D))
(connected aetr_bin ati_bin if age_bin == 2 & aetr_bin < 0.5, msize(small) mcolor(orange) lcolor(organge) msymbol(S))
(connected aetr_bin ati_bin if age_bin == 3 & aetr_bin < 0.5, msize(small) mcolor(green) lcolor(green) msymbol(T))
(connected aetr_bin ati_bin if age_bin == 4 & aetr_bin < 0.5, msize(small) mcolor(purple) lcolor(purple) msymbol(X)),
	title("AETR as a function of ATI") 
	xtitle("ATI range") 
	ytitle("AETR")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) */
	ylabel(-0.2(0.1)0.5,grid)
	legend(label(1 "Age < 25"))  
	legend(label(2 "Age 25-34")) 
	legend(label(3 "Age 35-44"))
	legend(label(4 "Age 45-54"))  
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`graphpath'/aetr_ati_bin_ages.pdf", replace;





capture log close ;


stop running ;

/* loop over ages, creating scatter plot */
forvalues i = 21/81 { ;
	twoway (scatter aetr ati if aetr < max_allowable_aetr & aetr > min_allowable_aetr & ati > 5 & ati < 500000 , msize(small) mcolor(blue) lcolor(blue) msymbol(o)),
		title("AETR vs. ATI, primary filer age = `i'") 
		xtitle("Adjusted Total Income") 
		ytitle("AETR")
		/*xscale(range(1 8)) 
		xlabel(1(1)8) 
		ylabel(-4(4)18,grid)*/
		scheme(s1mono) 
		saving(graph1, replace); 
	graph export "`graphpath'/ATI_level/aetr_ati_age`i'.pdf", replace;
	
	
	twoway (scatter aetr ln_ati if aetr < max_allowable_aetr & aetr > min_allowable_aetr & ati > 3500 & ati < 500000 , msize(small) mcolor(green) lcolor(green) msymbol(o)),
		title("AETR vs. ln(ATI), primary filer age = `i'") 
		xtitle("ln(Adjusted Total Income)") 
		ytitle("AETR")
		/*xscale(range(1 8)) 
		xlabel(1(1)8) 
		ylabel(-4(4)18,grid)*/
		scheme(s1mono) 
		saving(graph2, replace); 
	graph export "`graphpath'/ATI_log/aetr_ln_ati_age`i'.pdf", replace;

} ;