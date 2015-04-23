************************************************************************************   
/*Calibration of Household Consumption Parameters in the OLG Dynamic Scoring Model*/
************************************************************************************


							**********************
							*****SECTION 1.1******
							**********************
						
*********************
**Interview Surveys**
*********************						
						
                 *****************************************
                  *STEP 1 : Acquire the Consumption Data*
                 *****************************************
				 
/*Download the Consumer Expenditure Survey (CEX), years 2000-2013. 
You can find the data here: http://www.bls.gov/cex/pumdhome.htm
 
Since we want to use Stata to manipulate the data, download the Stata data 
files including the codebook(s) that accompany the data.*/

/* Use the monthly expenditure and income files (MTBI & ITBI) of the Interview Survey.
Also, use the FMLI interview files for demographic data.*/


                 *****************************************
                         *STEP 2: Format the Data*
				 *****************************************
/* Once each year's file is downloaded, we create a loop to append them together into one pooled 
cross-sectional dataset. Total Household Income (UCC 980000)is available in the ITBI files.
 Since the consumption, income and demographic data is found in different files 
 we will have to merge the 3 appended files to create one comprehensive dataset. */ 

clear
set more off
*log close

log using "E:\datasets\Tatenda\OPSC\Programs\Calibration4_12.smcl", replace

global path "E:/datasets/Tatenda/OPSC/Data"

local counter = 0
local year "00 01 02 03 04 05 06 07 08 09 10 11 12 13"
local Qtr "1x 2 3 4 1"

foreach yr of local year  {
local counter2 = 0
	foreach x of local Qtr  {

	if `yr' < 13 & "`x'" == "1" {
		di "Skip"
		}
	else {
			if `counter2' == 4 {
				local yr2 = `yr'+1
				}
			else {
				local yr2 = `yr'
				}	
			if `yr2' < 10 {
					local yr2 = "0"+"`yr2'"
					}
			else {
					local yr2 = "`yr2'"
					}		

		di "`x' `yr' `counter' `counter2'"
		
		/* Clean up income data files*/			
				local dir = "${path}/Raw/20`yr'/intrvw`yr'/itbi`yr2'"+"`x'"+".dta"
				use `dir', clear
				di "`dir'"
				foreach var of varlist * {
					rename `var' `var'_inc
					}
				rename refmo_inc ref_mo
				rename refyr_inc ref_yr
				rename newid_inc newid 
				tostring newid, replace
				gen cu = substr(newid,1,5) if length(newid) == 6
				replace cu = substr(newid,1,6) if length(newid) == 7
				gen no_intrvw = substr(newid,-1,1)
				keep if ucc_inc == "980000" | ucc_inc== "900150" | ucc_inc== "980230" | ucc_inc== "980240" | ucc_inc== "980260" 
				*ucc_inc 980000 is income before tax and 900150 is annual value of food stamps
				*ucc_inc 980230 , 980240 & 980260 are Homeowner variables  distinguishing whether the consumer unit owns (with or without mortgage) or rents the home.
				drop value__inc
				reshape wide value_inc, i(cu ref_mo) j(ucc_inc) string
				if `counter' == 0 {
						save "${path}/Intermediate/income.dta", replace
						}
					else {
						append using "${path}/Intermediate/income.dta"
						save "${path}/Intermediate/income.dta", replace
						}
			
		/* Clean up demographic data files*/				
				local dir = "${path}/Raw/20`yr'/intrvw`yr'/fmli`yr2'"+"`x'"+".dta"
				use `dir', clear
				di "`dir'"
				tostring newid, replace
				gen cu = substr(newid,1,5) if length(newid) == 6
				replace cu = substr(newid,1,6) if length(newid) == 7
				gen no_intrvw = substr(newid,-1,1)
				keep cu no_intrvw finlwt21 age_ref age_ref_ no_earnr fam_size fam__ize roomsq roomsq_ bathrmq bathrmq_ hlfbathq
				* Age of Reference Person (AGE_REF), Household size (FAM_SIZE), dwelling size (ROOMSQ+BATHRMQ+HLFBATHQ), number of earners in household (NO_EARNR) and FINLWT21 is the weighting variable for the full sample
					if `counter' == 0 {
						save "${path}/Intermediate/demo.dta", replace
						}
					else {
						append using "${path}/Intermediate/demo.dta"
						save "${path}/Intermediate/demo.dta", replace
						}
						
		/* Clean up consumption data files*/
				local dir = "${path}/Raw/20`yr'/intrvw`yr'/mtbi`yr2'"+"`x'"+".dta"
				use `dir', clear
				di "`dir'"
				tostring newid, replace
				gen cu = substr(newid,1,5) if length(newid) == 6
				replace cu = substr(newid,1,6) if length(newid) == 7
				gen no_intrvw = substr(newid,-1,1)
					if `counter' == 0 {
						save "${path}/Intermediate/consumption.dta", replace
						}
					else {
						append using "${path}/Intermediate/consumption.dta"
						save "${path}/Intermediate/consumption.dta", replace
						}
				local counter = `counter'+1
				}
		local counter2 = `counter2'+1
	}
}

  *Merging files*
  ***************

 /*First, merge the consumption and income data by Consumer Unit, Reference month & Reference Year. 
 Then merge the consumption/income data with demographic data by CU and Interview Number */
*For glossary of Variable names and CEX survey terminologies, see the one of the User documentation files that is available in the downloaded annual files.
		* Pg. 107 http://www.bls.gov/cex/2013/csxintvw.pdf 

*use "${path}/Intermediate/consumption.dta", clear
    
merge m:1 cu ref_mo ref_yr using "${path}/Intermediate/income.dta"
rename _merge income_mrg
  
merge m:1 cu no_intrvw using "${path}/Intermediate/demo.dta"
rename _merge demo_mrg

destring ref_yr ref_mo no_intrvw, replace

codebook cu 
/*After merging the files we have all the waves of data that are present in the 
interview years between 1999Q4-2014Q1. This leaves us with observations of 
141807 households for as many as four quarters each.*/


***** Annual Consumption Amounts*******

by cu ucc, sort: egen cost_yr = sum(cost)
/*This gives you the amount spent by the household (CU) for a particular consumption item during their respective interview year.
Note that this is not the total annual consumption amount by consumption category, its by UCC. */

save "${path}/Intermediate/combined.dta", replace
   
								**********************
								*****SECTION 1.2******
								**********************  
 
********************
**Sample Selection**
********************

/* "During any quarter, about 25 prcent of the respondents are interviewed about 
their expenditures for the first time, and this group is then interviewed for 3 additional quarters." See- Fullerton & Rogers pg 131
*We use two waves of data. For example, one wave includes quarterly surveys from 2012Q1-2012Q4 
(thus covering expenditures made from 2011Q4-2012Q3). The second includes surveys from 2012Q2-2013Q1 (covering expenditures from 2012Q1-2012Q4). */

*Wave 1 Q4-Q3

*Wave 1: YearQ1*
gen wave=.
forvalues x= 10/12{
by cu no_intrvw ref_mo, sort: replace wave= 11 if ref_mo ==`x' & no_intrvw==2
}

*Wave 1: YearQ2*
forvalues x= 1/3{
by cu no_intrvw ref_mo, sort: replace wave= 12 if ref_mo ==`x' & no_intrvw==3
}

*Wave 1: YearQ3*
forvalues x= 4/6{
by cu no_intrvw ref_mo, sort: replace wave= 13 if ref_mo ==`x' & no_intrvw==4
}

*Wave 1: YearQ4*
forvalues x= 7/9{
by cu no_intrvw ref_mo, sort: replace wave= 14 if ref_mo ==`x' & no_intrvw==5
}

********
*Wave 2 Q1-Q4

*Wave 2: YearQ1*
forvalues x= 1/3{
by cu no_intrvw ref_mo, sort: replace wave= 21 if ref_mo ==`x' & no_intrvw==2
}

*Wave 2: YearQ2*
forvalues x= 4/6{
by cu no_intrvw ref_mo, sort: replace wave= 22 if ref_mo ==`x' & no_intrvw==3
}

*Wave 2: YearQ3*
forvalues x= 7/9{
by cu no_intrvw ref_mo, sort: replace wave= 23 if ref_mo ==`x' & no_intrvw==4
}

*Wave 2: YearQ4*
forvalues x= 10/12{
by cu no_intrvw ref_mo, sort: replace wave= 24 if ref_mo ==`x' & no_intrvw==5
}
tab ref_mo no_intrvw if wave==.
drop if wave==.
codebook cu
*After adjusting for the two waves, we have XXXX households for as many as four quarters each. 
gen wave_yr = .

foreach x in 11 12 13 14 21 22 23{
	replace wave_yr= ref_yr if wave==`x'
}
replace wave_yr = wave_yr[_n-1] if wave_yr==.

**************
/*From the two waves of surveys we use, we drop respondents who participate in the survey
for less than a year (i.e., have less than four quarters of data).*/

*To do this we flag the consumer units that dont have less than 4 interviews and we drop them.
by cu no_intrvw, sort: gen dup_intrvw= _n
by cu no_intrvw ref_yr, sort : gen sum_intrvw = sum(dup_intrvw) if dup_intrvw==1
by cu sum_intrvw, sort : egen cusum_intrvw = sum(sum_intrvw)
gsort cu -cusum_intrvw
 replace cusum_intrvw = cusum_intrvw[_n-1] if sum_intrvw == . & _n != 1
codebook cu if cusum_intrvw <4
drop if cusum_intrvw <4 

**************
/*We further exclude from our sample respondents who received food stamps and those 
with incomplete income reporting (i.e., for whom we do not observe reported amounts of total income). */

* filter the data browser by value_inc980000 == . to see if they are any cu's with mising income values.
drop if value_inc900150 != .
drop value_inc900150

tab ucc if income_mrg==1
* Most of these UCC items will be dropped in the Consumption Category creation section of this file. These items are classified as assets.
* However for now so that they do not get dropped as having incomplete income reporting, populate the missing income values with the monthly income noted in other observations relating to the particular CU and interview number.
gsort cu no_intrvw -value_inc98000
 by cu no_intrvw: replace value_inc98000 =value_inc98000[_n-1] if income_mrg== 1 & _n != 1

****************
/*Some missing income values from income_mrg==1 are still missing because the CU did not report any income for that interview period.
Hence, I chose to drop these items with other CU's that do not have a single income value for any one of the 4 interviews*/

by cu no_intrvw value_inc98000, sort: gen incomp_increp= 1 if value_inc98000 ==.
gsort cu -incomp_increp
 by cu: replace incomp_increp = incomp_increp[_n-1] if _n != 1
codebook cu if incomp_increp == 1 
*There are 5266  households with incomplete income reports which will be dropped from our sample
drop if incomp_increp == 1
tab cu if value_inc980000==.
codebook cu
*All the above exclusions leave us with  41025 households.

save "${path}/Intermediate/combined.dta", replace


********************************************************************************
 *Deflating data*
******************
  
/* Next, we put all the dollar amounts in constant, year 2013
dollars. Do this using the consumer price index (CPI). You can find the CPI 
deflators here: http://research.stlouisfed.org/fred2/series/CPIAUCSL (using the 
deflator for all items is fine for now). */

*Use the 'freduse' Stata plug-in instead of importing the data through excel
   
freduse CPIAUCSL, clear
gen ref_yr= year(daten)
gen ref_mo= month(daten)
rename CPIAUCSL cpi
drop if ref_yr < 1999
gen pindex= cpi/234.594
save"${path}/Intermediate/pindex.dta", replace

merge 1:m ref_mo ref_yr using "${path}/Intermediate/combined.dta"
rename _merge pi_mrg
drop if pi_mrg==1
rename value_inc980000 value_inc
gen cost_wcpi = cost/pindex
gen value_incwcpi = value_inc/pindex
save "${path}/Intermediate/combined_wcpi.dta", replace

********************************************************************************									
		
sum value_incwcpi
/*Note: This dataset is still comprised of monthly statistics, file could be collapsed to annual figures only.
However, current after the cleaning and sample selection, we have a total number of 10905554 observations, across all monthly consumption items. */
 
misstable summ
*Note that room_sq and other dwelling size variables have some missing observations. Some households did not report their dwelling size.

codebook cu
*Number of Households = 41128  

save "${path}/Intermediate/combined_wcpi.dta", replace

								**********************
								*****SECTION 1.3******
								**********************
									
						*********************************************
						***STEP 3: Creating Consumption Categories***
						*********************************************
 
 **************************************
 *Creating Consumption Good Categories*
***************************************

*Use dummy variables to create each of the 17 categories
foreach categ_list in food1_dum alcohol2_dum tobacco3_dum hfu4_dum shelter5_dum furn6_dum appli7_dum app8_dum pubt9_dum cars10_dum cashpers11_dum finserv12_dum readent13_dum houseop14_dum gasoil15_dum healthcare16_dum edu17_dum {
	gen `categ_list' = 0
}

/* Create a loop for each category inorder to assign a 1 to each dummy variable for each 
all respective UCC's in each category as given in the 2013 Istub files. 

Aggregations are as follows:

** Table 1 **	
***************************************************************************************
Category   *Description 	   	   *CEX variables 		*Additional categories 
														from "Miscellaneous"
***************************************************************************************														
1 			Food 					FOODTOTL
2 			Alcohol 				ALCBEVG
3 			Tobacco 				TOBACCO
4 			Household fuels and 
			utilities 				UTILS
5 			Shelter 				910050 (for homeowner)
									RNTDWELL (for renter)
6 			Furnishings 			HHFURNSH - MAJAPPL
7 			Appliances		 		MAJAPPL
8 			Apparel 				APPAREL
9 			Public transport		PUBTRANS
10 			New and used cares, 	VEHPURCH + VEHOTHXP
			fees, and maintenance
11 			Cash contributions and 	
			personal care 			PERSCARE + CASHCONT 	+680140+680901													
			(personal services)
12 			Financial services 		INSPENSN 				680210+680220+680902
															+710110+005420+005520
															+005620+880210+620112
13 			Reading & entertainment 
			(recreation) 			READING + ENTRTAIN 		+680904+680905+790600+620926
14 			Household operations 
			(nondurables) 			HHOPER 					+620115+900002+680110
15 			Gasoline and motor oil 	GASOIL
16 			Health care 			HEALTH
17 			Education 				EDUCATN
***************************************************************************************

*NB. The above table is only based on the 2013 IStub file. CEX Istub files list 
the breakdown of all aggregations found in CEX interview data for a given year.
We will exclude items noted in the files as "Assets" or "Addenda."  
Also note that consumption codes change from year to year, some are added, some
 are removed and some are amended so one must check throughly to ensure all 
 relevant consumption codes have been added to each category. More items not found
 in this file but available in the 2000-2013 file were later added to the respective 
 categories. */
 
foreach ucc_list in 790240  190904  790410  190901  190902  190903  790430  800700 {
	replace food1_dum = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 790330  790420  200900 {	
	replace alcohol2_dum = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 630110   630210 {
	replace tobacco3_dum = 1 if ucc == "`ucc_list'"
	}
	

foreach ucc_list in 260211  260212 260213  260214 260111  260112  260113  260114 250111 250112 250113  250114  250911  250912   250913   250914   250211   250212   250213   250214   270101  270102 270104 270105  270106   270211  270212  270213  270214   270411    270412  270413    270414      270901 270902  270903 270904{
	replace hfu4_dum = 1 if ucc == "`ucc_list'"
}

******
					***************************
					**Section 1.4.2 : Shelter**
					***************************
foreach ucc_list in 910050 210110 800710 350110 320624 230150  230121   230141 240111  240121  240211   240221   240311  240321  320611 990920  790690 320621  320631  {
	replace shelter5_dum = 1 if ucc == "`ucc_list'"
}
* Add UCC 910050 to take into account adjustments on shelter.

*******

foreach ucc_list in 280110 280120 280130   280210 280220 280230 280900 280140 290110 290120   290210 290310 290320   290410   290420   290430   290440 320111 320310 320320   320330 320345 320340 320350 320360 320370 320521 320522 320120   320130 320150 320220 320221 320233 320232 320410 320420 320901 320902 320903 320904 340904  30130 690111 690117 690119 690120 690115 690116 690210   690230   690242   690241   690243 690245 690244{
	replace furn6_dum = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in  230117 230118   300111 300112 300216 300217 300211 300212 300221 300222 300311 300312   300321   300322 300331 300332 300411 300412   320511   320512 {
	replace appli7_dum = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 360110 360120  360210  360311  360312  360320  360330 360340  360350  360410    360420 360513  360901  360902  370110    370125   370120  370130  370211  370212 370213 370220  370311   370314  370903  370904   370902  380110  380210  380311 380312 380313 380315 380320  380333  380340  380410   380420  380430  380510  380901   380902  380903  390110  390120 390210  390223  390230  390310  390321 390322  390901   390902  410110  410120 410130 410140  410901 400110 400210  400310 400220  420110 420120 420115 430110  430120  440110  440120  440130 440140 440150 440210  440900 {
	replace app8_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 530110  530210   530311  530312    530411  530412  530510  530901  530902  {
	replace pubt9_dum  = 1 if ucc == "`ucc_list'"
}
	
foreach ucc_list in 450110   450210    460110  460901    450220     450900    460902   460903    510110    510901     510902      850300      470220       480110     480212       480216    480213       480214     480215    490110      490300      490211      490212        490221     490231      490232       490311       490312   490313     490314    490318     490411   490412     490413    490900      490501  490319   500110   520511    520512     520521     520522     520902      520905    520905    450350   450353    450354    450351     450352     450313     450314   450410    450413    450414  520310   520410    520531     520532     520541     520542  520550    520560  620113   620114{
	replace cars10_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 800804 800111  800121 800811  800821  800831  800841   800851   800861 640130  640420  650310 680140 680901 {
	replace cashpers11_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 680210 680220 680902 710110 005420 005520 005620 880210 620112 700110 002120 800910 800920 800931 800932 800940{
	replace finserv12_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in   620926 680904 680905 790600 590310 590410 590220 590230 660310 690118 610900 620111 620121 620122   620211 620213 620214  620212 620221   620222 620310 620903   310316  310140  270310 270311 620930   310210 310220   310231 310232 310240 310400 340610  340902 310311  310313 310314  310320  310340  310350  340905  610130  620904  620912  620917  620918  310333  310334  690320  690330  690340  690350  610320  620410  620420  610110  610140  610120  600121  600122  600141  600142  600132  520904  520907  620909  620919  620906  620921  620922  600110  520901  600210  600310  600410  600420  600430  600901  600902  620908  610210  620330  620905  610230  620320  680310  680320{             
  replace readent13_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 680110   900002   620115   340211  340212  340210 340906 340910   670310   340310 340410  340420 340520  340530  340914 340915 340903  330511   340510  340620  340630  340901 340907  340908  690113  690114  690310 990900{
  replace houseop14_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 470111 470112  470113  470211 470212{
	replace gasoil15_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 580111  580113  580112  580114 580115 580116  580312 580904 580906 580311 580400 580903 580905  560110  560210  560310 560400  560330 570111  570220 570230  540000   550110  550320  550330  550340  640430 570901 570903{
	replace healthcare16_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 670110 670210  670410  670903  670901 670902  660110   660210 660410  660901  660902{
	replace edu17_dum = 1 if ucc == "`ucc_list'"
}

 *Cleaning up additional UCC's that are not covered in codes for the 17 consumption categories above*
****************************************************************************************************
    
  /*First identify any items that are not assigned a 1 in any of the 17 dummies we created.
  We then create an excel spreadsheet to see which years these UCC's are from since we are currently only using the 2013 Istub file.
  We also identified which of these items were flagged as an asset, addenda or expenditure 
  in order to also narrow the area of search. */
  
gen missing_cat = 0
foreach x in food1_dum alcohol2_dum tobacco3_dum hfu4_dum shelter5_dum furn6_dum appli7_dum app8_dum pubt9_dum cars10_dum cashpers11_dum finserv12_dum readent13_dum houseop14_dum gasoil15_dum healthcare16_dum edu17_dum {
	replace missing_cat = 1 if `x' == 1
}
  
tab ucc ref_yr if missing_cat !=1 
*export results into excel spreadsheet and use 2000-2013 Istub files to match titles with missing UCC codes. 
*determine whether item is an expenditure and troubleshoot any other items that may be missing through searching user documentations and data dictionaries provided on the BLS CEX webpage

/* We drop all UCC's that are flagged as Assets or Addenda. These items can be found in the data descriptive files as amendments to the data 
 or as part of a list that we have not fully accounted for why they are not included in the Isub files */
 
foreach misucc_list in 006001 006002 006003	006004	006005	006006	006007	006008	006009	006010	006011	006012	006013	006014	220512	220513	220611	220612	220615 220616 430130	450216	450226	450310	450311	450312	450411	450412	460116	460907	460908 450116	600127	600128	600138	600143	600144	790210  790610	790611	790620	790630	790640	790710	790730	790810	790830	790910	790920	790930	790940	790950	800721	800803	810101	810102	810301	810302	810400	820101	820102	820301	820302	830101	830102	830201	830202	830203	830204	830301	830302	830303	830304	840101	840102	850100	850200	860100	860200	860301	860302	860400	860500	860600	860700	870101	870102	870103	870104	870201	870202	870203	870204	870301	870302	870303	870304	870401	870402	870403	870404	870501	870502	870503	870504	870605	870606	870607	870608	870701	870702	870703	870704	870801	870802	870803	870804	880120	880220	880320	910042	910050	910100	910101	910102	910103	910104	910105	910106	910107	950024	950030	950031	990950{
	drop if ucc== "`misucc_list'"
}
*Due to adjustments on Shelter in new Calibration guide version Section 1.4.2, we drop the following UCC's that fall under SHELTER in the 2013 ISTUB file
foreach shelterucc_list in 220311   220313   880110  220211  220121  210901  320625  230112 230113 230114 230115    230151 230122  230142  240112 240122  240312  240322 320622   240213  240212 240222  320632   320612  990930  230901   340911  220901 220312 220314 880310 220212 320626  220122  210902  230152 230123   240113  240123 240214 240223 240313 240323   320613  320623 990940   320633  230902  220902  340912  210310   210210 {  
	drop if ucc== "`shelterucc_list'"
	}
tab ucc if missing_cat !=1 

/*The following are loops that amend some of the categories that had consumption items 
that were not captured by previous categorization due to a number of possible reasons mentioned earlier in this section. */

foreach ucc_list in  790220 790230{
	replace food1_dum = 1 if ucc == "`ucc_list'"
}
	
foreach ucc_list in 790310 790320 {	
	replace alcohol2_dum = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 250221 250222 250901	250902	250903	250904	270103{
	replace hfu4_dum = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 220321{
	replace shelter5_dum = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 230133 230134 320163 320210	320231 690112 690220{
	replace furn6_dum = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 360511	360512	370312	370313	380331	380332	390221	390222{
	replace app8_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 490502 520110 520111 520112 520516 520517 520903 520906{
	replace cars10_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 310110	310120	310130	310230	310341	310342 590111 590112 590211  590212   620916{
	replace readent13_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in P570110 570210 570240 580901 580907{
	replace healthcare16_dum  = 1 if ucc == "`ucc_list'"
}

foreach ucc_list in 660900{
	replace edu17_dum = 1 if ucc == "`ucc_list'"
}

foreach x in food1_dum alcohol2_dum tobacco3_dum hfu4_dum shelter5_dum furn6_dum appli7_dum app8_dum pubt9_dum cars10_dum cashpers11_dum finserv12_dum readent13_dum houseop14_dum gasoil15_dum healthcare16_dum edu17_dum {
	replace missing_cat = 1 if `x' == 1
}	
tab ucc ref_yr if missing_cat !=1 
sum value_incwcpi
* Total number of observations is now:   23577969 

save "${path}/Intermediate/categ_combo_wcpi.dta", replace

****Generate a category variable that assigns a numerical group value to each consumption cost observation.
gen category =1 if food1_dum == 1
replace category =2 if alcohol2_dum == 1
replace category =3 if tobacco3_dum == 1
replace category =4 if hfu4_dum == 1
replace category =5 if shelter5_dum ==1
replace category =6 if furn6_dum ==1
replace category =7 if appli7_dum == 1
replace category =8 if app8_dum == 1
replace category =9 if pubt9_dum == 1
replace category =10 if cars10_dum == 1
replace category =11 if cashpers11_dum == 1
replace category =12 if finserv12_dum == 1
replace category =13 if readent13_dum == 1
replace category =14 if houseop14_dum ==1
replace category =15 if gasoil15_dum == 1
replace category =16 if healthcare16_dum == 1
replace category =17 if edu17_dum ==1

*Create income percentile*
**************************
*Used this variable to make graphs noted in the instruction of the first version of the Calibration Guide
xtile incpct = value_incwcpi, n(10)
tab incpct, gen(iq)


save "${path}/Intermediate/categorizedcex.dta", replace

						**********************
						*****SECTION 1.4******
						**********************

*****************************************
**Adjustments for Durables and Shelter**
*****************************************

***Create Income categories for Adjustments to Durables and Shelter
sum value_incwcpi
  * there are  8314698 observations left in the dataset
tab value_incwcpi if (value_incwcpi >=0 & value_incwcpi< 1)
 * there are   1,142   observations with income that are between 0 and 1. 
tab value_incwcpi if value_incwcpi< 1
 * there are   17,500  observations with income that are less tha between 0 and 1. 
drop if value_incwcpi< 1

by ref_yr cu ref_mo, sort: gen dup_inc = _n
by ref_yr cu, sort: egen avg_mo_inc = mean(value_incwcpi) if dup_inc == 1
gen avg_yr_inc = 12* avg_mo_inc
by ref_yr cu, sort: replace avg_yr_inc = avg_yr_inc[_n-1] if avg_yr_inc == .
   
gen income_cat =0
replace income_cat= 1 if (value_incwcpi >=1 & value_incwcpi < 5000)
replace income_cat= 2 if value_incwcpi >= 5000 & value_incwcpi<10000
replace income_cat= 3 if value_incwcpi >= 10000 & value_incwcpi <15000
replace income_cat= 4 if value_incwcpi >= 15000 & value_incwcpi <20000
replace income_cat= 5 if value_incwcpi >= 20000 & value_incwcpi <25000
replace income_cat= 6 if value_incwcpi >= 25000 & value_incwcpi <30000
replace income_cat= 7 if value_incwcpi >= 30000 & value_incwcpi <40000
replace income_cat= 8 if value_incwcpi >= 40000 & value_incwcpi <50000
replace income_cat= 9 if value_incwcpi >= 50000 & value_incwcpi <75000
replace income_cat= 10 if value_incwcpi >= 75000 & value_incwcpi <100000
replace income_cat= 11 if value_incwcpi >= 100000
* Note that after the sample selection, the dataset does not have any consumer units that earn 100000 or more annually.
tab  value_incwcpi if income_cat ==0

***Create Age Categories for CEX summary tables to be created in the next section
gen age_cat =0
replace age_cat = 1 if (age_ref >=20 & age_ref < 25)
replace age_cat = 2 if age_ref >=25 & age_ref < 30
replace age_cat= 3 if age_ref >=30 & age_ref < 35
replace age_cat= 4 if age_ref >=35 & age_ref < 40
replace age_cat= 5 if age_ref >=40 & age_ref < 45
replace age_cat= 6 if age_ref >45 & age_ref < 50
replace age_cat= 7 if age_ref >=50 & age_ref < 55
replace age_cat= 8 if age_ref >=55 & age_ref < 60
replace age_cat= 9 if age_ref >=60 & age_ref < 65
replace age_cat= 10 if age_ref >=65 & age_ref < 70
replace age_cat= 11 if age_ref >=70 & age_ref < 75
replace age_cat= 12 if age_ref >=75
tab  age_ref if age_cat ==0
							
***Create Household type categories required for the three tables

*Family Size
gen famsize_cat = 0
replace famsize_cat = 1 if fam_size < 3
replace famsize_cat = 2 if fam_size >= 3 & fam_size <=4
replace famsize_cat = 3 if fam_size > 4
tab  fam_size if famsize_cat ==0

*Tenure

/*Homeown includes information on whether the head of household owns or rents the home.
 For those who own the house they are further divided into Owners with or owners without a mortgage */
gen tenure = .
tostring tenure, replace
replace tenure = "Owner" if value_inc980230 != . |  value_inc980240 != .
replace tenure = "Renter" if  value_inc980260 != .
tab tenure if tenure=="."

*Number of Earners
tab no_earnr
gen noearner_cat =0
replace noearner_cat = 1 if no_earnr <1
replace noearner_cat = 2 if no_earnr ==1
replace noearner_cat = 3 if no_earnr >=2
tab no_earnr if noearner_cat ==0

*Dwelling Size
gen dwellingsize = (roomsq + bathrmq + hlfbathq)
br
gen dwellingsize_cat = 0
replace dwellingsize_cat = 1 if dwellingsize < 4 
replace dwellingsize_cat = 2 if dwellingsize >= 4 & dwellingsize <=5
replace dwellingsize_cat = 3 if dwellingsize > 5
tab dwellingsize if dwellingsize_cat ==0

**** Check data for missing values and unnecesssary variables.****
******************************************************************
 
misstable summ
*Note that room_sq and other dwelling size variables have some missing observations. Some households did not report their dwelling size.

/* Note the missing observations
                                                          Obs<.
                                                +------------------------------
               |                                | Unique
      Variable |     Obs=.     Obs>.     Obs<.  | values        Min         Max

	   bathrmq |    52,151             8245047  |     10          0          10
      hlfbathq |    88,922             8208276  |      7          0           9
        roomsq |    56,428             8240770  |     29          0          99
  dwellingsize |    96,626             8200572  |     35          1         10319
*/

* Browse data dictonary that is published with each year's data to find the descriptions for the following items that are being dropped.
drop date daten seqno alcno cost_ rtype gift uccseq pubflag pubflag_inc gift_inc age_ref_ bathrmq_ fam__ize roomsq_

codebook cu

*Number of Households = 41120  

**Create label names**
**********************
* It is helpful to create label names especially for display (graphing) purposes.

label var ref_yr "Reference Year"
label var ref_mo "Reference Month"
label var ucc "Universal Clssification Code"
label var cu "Consumer Unit"
label var cost "Consumption Cost"
label var no_intrvw "interview Number"
label var value_inc "Income value"
label var age_ref "Age of Reference Person"
label var pindex "CPI all items in 2013 dollars"
label var cost_wcpi "Consumption Cost adjusted for inflation"
label var value_incwcpi "Income value adjusted for inflation"
label var finlwt21 "Weight for population"
label var famsize_cat "Household Size"
label var noearner_cat "Number of Earners in Household"
label var tenure "Tenure - Owner/Renter"
label var dwellingsize_cat "Dwelling Size"
label var age_cat "Age Category"
label var income_cat "Income Category"
label var category "Consumption Categories"

save "${path}/Intermediate/adjustedconsumption.dta", replace

							******************
							**1.4.1 Durables**
							******************
**********							
**Tables**
**********

/*The BLS survey data that is published does not include values for those who spent $0 on an item 
or for those who failed to report their expenditure for any given item. This affects our mean calculations as some samples
have very few observations which gives us an inaccurate representation of the sample. In order to mitigate this problem,
divide the total cost per consumption category in any given year by the total number of observations surveyed in any given year accross all categories.
So for example, if category 17 only has 10000 people who reported their income, we divide the sum of all expenditure on education by the total number of 
people surveyed (full sample) which in this case is approximately 41000 instead of dividing by 10000. */

								***Table 2***
								

*Adjustments to Appliances*
****************************

bysort cu income_cat famsize_cat  wave_yr: gen first_obsfam = 1 if _n == 1
*this helps us flag each household once, if it falls under the specified household characteristics.

gen appliances_adj= .
*forvalues y=2000/2013{
forvalues y = 2000/2013{
	forvalues i = 1/10 {
		forvalues n = 1/3 {
		quietly summarize cost_wcpi if appli7_dum == 1 & income_cat == `i' & famsize_cat == `n' & wave_yr == `y', detail
		local total_exp = r(sum) 
		di `total_exp' 
		di "Inc_"`i' "Fam_"`n' 
		quietly summarize first_obsfam if income_cat == `i' & famsize_cat == `n' & wave_yr == `y', detail
		local total_people = r(sum)
		di `total_people'
		di "Inc_"`i'  "Fam_"`n'
		replace appliances_adj = `total_exp'/`total_people' if appli7_dum == 1  & income_cat == `i' & famsize_cat == `n' & wave_yr == `y'
		di appliances_adj
		di "Inc_"`i'  "Fam_"`n' "Wave_yr_"`y' "next"
		}

	}
}
/* this loop sums the appliance expenditures by year and household characteristics 
then sums the number of unique consumer units that meet the same household characteristics.
Divide those two sums to get the average annual expenditures  on Appliances 
which includes non-reporting or zero expenditure housholds. */

*Average Annual Expenditures on Appliances, by Income and Family Size (2011-2012)
table income_cat famsize_cat if wave_yr ==2012, c(mean appliances_adj)
*Average Annual Expenditures on Appliances, by Income and Family Size (2012-2013)
table income_cat famsize_cat if wave_yr ==2013, c(mean appliances_adj)

						    ***Table 3***

	 *****We repeat the same procedure for adjustments to Furniture*****

*Adjustments to Furniture*
****************************
bysort cu income_cat dwellingsize_cat tenure wave_yr: gen first_obsdwel = 1 if _n == 1

gen furniture_adj= .
forvalues y = 2010/2013{
	foreach t in Owner Renter {
		forvalues i = 1/10 {
			forvalues n = 1/3 {
				quietly {
					summarize cost_wcpi if furn6_dum == 1 & tenure== "`t'" & income_cat == `i' & dwellingsize_cat == `n' & wave_yr == `y', detail
					local total_exp = r(sum) 
					summarize first_obsdwel if tenure== "`t'" & income_cat == `i' & dwellingsize_cat == `n' & wave_yr == `y', detail
					local total_people = r(sum)
					replace furniture_adj = `total_exp'/`total_people' if furn6_dum == 1 & tenure== "`t'"  & income_cat == `i' & dwellingsize_cat == `n' & wave_yr == `y'
				}
			}
		}
	}
}
*Average Annual Expenditures on Furniture, by Income, Tenure and Dwelling Size (2012)
table income_cat dwellingsize_cat if wave_yr ==2012, by(tenure) c(mean furniture_adj)		
*Average Annual Expenditures on Furniture, by Income, Tenure and Dwelling Size (2013)
table income_cat dwellingsize_cat if wave_yr ==2013, by(tenure) c(mean furniture_adj)		
	
							 ***Table 4*** 

* New Categorization for Motor Vehicle related expenditures.*

/*NB.We impute the flow of consumption from motor vehicles by averaging annual expenditures
on the durable component of motor vehicles (CEX variable VEHPURCH) by income group and number of earners */

gen motorveh_dum=0
foreach ucc_list in 450110   450210    460110  460901    450220     450900    460902   460903  {
	replace motorveh_dum  = 1 if ucc == "`ucc_list'"
}

*Adjustment to Motor Vehicles*
********************************
bysort cu income_cat noearner_cat wave_yr: gen first_obsearn = 1 if _n == 1

gen motorveh_adj= .
forvalues y = 2000/2013{
	forvalues i = 1/10 {
		forvalues n = 1/3 {
			quietly {
				summarize cost_wcpi if motorveh_dum == 1  & income_cat == `i' & noearner_cat == `n' & wave_yr == `y', detail
				local total_exp = r(sum) 
				summarize first_obsearn if income_cat == `i' & noearner_cat == `n' & wave_yr == `y', detail
				local total_people = r(sum)
				replace motorveh_adj = `total_exp'/`total_people' if motorveh_dum == 1  & income_cat == `i' & noearner_cat == `n' & wave_yr == `y'
			}
		}
	}
}

*Average Annual Expenditures on Motor Vehicles, by Income and Number of Earners (2012)	
table income_cat noearner_cat if wave_yr ==2012, c(mean motorveh_adj) 
*Average Annual Expenditures on Motor Vehicles, by Income and Number of Earners (2013)	
table income_cat noearner_cat if wave_yr ==2013, c(mean motorveh_adj)
 
							*************
							*CEX SUMMARY*
							*************
					**Mean Annual Expenditures from 2000-2013 **
					
/*Not sure if these averages are correct. May need to add wave_yr in the bysort
 and in the loop like we did in the adjustments above  
 */

*Full sample
************

bysort cu: gen first_obs = 1 if _n == 1 

gen mean_cat= .
forvalues y=2000/2013{
forvalues x=1/17 { 
		quietly summarize cost_wcpi if category == `x' & wave_yr == `y', detail 
		local total_cost = r(sum)
		di `total_cost' "cat_"`x'
		
		quietly summarize first_obs if wave_yr == `y', detail 
		local total_people = r(sum)
		di `total_people' 
				
		replace mean_cat = `total_cost'/`total_people' if category == `x' & wave_yr == `y'
		di mean_cat "cat_"`x'
	}
}
table category, c(mean mean_cat)
tab mean_cat if wave_yr ==2012

/*Alternatively, use the following code.
codebook cu
*Unique Values for Consumer Unit == 41120
by category, sort: egen sumcat= sum(cost_wcpi)
by category, sort: gen avgcatcost= sumcat/ 41120
table category, c(mean avgcatcost)
*/

		*Annual average for full sample for Adjusted Durables expenditures 
tab motorveh_adj if wave_yr ==2012
tab appliances_adj if wave_yr ==2012
tab furniture_adj if wave_yr ==2012
 
*By Income Category
*******************

bysort cu income_cat: gen firstinc_obs = 1 if _n == 1 

gen mean_inccat= .
forvalues y=2012/2013{
	forvalues x=1/17 { 
		forvalues i = 1/10 {
			quietly {
				summarize cost_wcpi if income_cat == `i' & category == `x' & wave_yr == `y', detail 
				local total_inc = r(sum) 
				summarize firstinc_obs if income_cat == `i' & wave_yr == `y', detail 
				local total_people = r(sum)
				replace mean_inccat = `total_inc'/`total_people' if income_cat == `i' & category == `x' & wave_yr == `y'
			}
		}
	}
}
table category income_cat if wave_yr ==2012, c(mean mean_inccat)
table category income_cat if wave_yr ==2013, c(mean mean_inccat)


*Mean for Adjusted Durables expenditures by Income Category
table income_cat, c(mean motorveh_adj) 
 table income_cat if wave_yr==2012, c(mean motorveh_adj) 
 table income_cat if wave_yr==2013, c(mean motorveh_adj)
 
 table income_cat if wave_yr ==2012, c(mean appliances_adj) 
 table income_cat if wave_yr ==2012, c(mean furniture_adj)
 
 
 ***stupid point this is happening because my adjustments are only created for 2012-2013
 /* this is an example of the change in annual averages if I include wave_yr in the code 
 vs when I dont specify wave year and average across all households annual consumption amounts.
 If we estimate these annually we may get less information for all categories in comparison to taking averages across all years*/
 table income_cat, c(mean furniture_adj)
 table income_cat, c(mean appliances_adj)
 
 
*By Age
*******
* what about people younger than 20, there are almost 20000 observations of expenditures made my 
bysort cu age_cat: gen firstage_obs = 1 if _n == 1 

gen mean_agecat= .
forvalues x=1/17 { 
	forvalues z = 1/12 {
		quietly {
			summarize cost_wcpi if age_cat == `z' & category == `x', detail 
			local total_age = r(sum)
			summarize firstage_obs if age_cat == `z' , detail 
			local total_people = r(sum)
			replace mean_agecat = `total_age'/`total_people' if age_cat == `z' & category == `x'
		}
	}
}	
table category age_cat, c(mean age_cat)

*Mean for Adjusted Durables expenditures by Age Category
 table age_cat if wave_yr ==2012, c(mean motorveh_adj)
 table age_cat if wave_yr ==2012, c(mean furniture_adj)
 table age_cat if wave_yr ==2012, c(mean appliances_adj)
 
save "${path}/Intermediate/adjustedconsumptionfinal.dta", replace

								**********************
								*****SECTION 1.5******
								**********************
									
					*********************************************
					** STEP 5: Estimate Consumption Parameters **
					*********************************************
					
******************
**Underreporting**
******************



***************************************************************************************************************************************************************************************************************************************************************************************
***************************************************************************************************************************************************************************************************************************************************************************************
*Previous Work

**********************************************OUTPUT TABLES AND GRAPHS**********************************************************

*****
log using "G:\Interns\Fall 2014 Interns\Tatenda K Mabikacheche\OPSC\OPSC\Data\Intermediate\Section1_4table_edit.smcl, replace"
**Tables**
**********

						   ***Table 2***

*Average Annual Expenditures on Appliances per Household by Income and Family Size (2012-2013)	
table income_cat famsize_cat if ref_yr ==2012 | ref_yr==2013, c(mean applicost_adj)


						    ***Table 3***

*Average Annual Expenditures on Furniture per Household by Income, Tenure and Dwelling Size (2012-2013)		
table income_cat dwellingsize_cat if ref_yr >=2012, by(tenure) c(mean furncost_adj)		
			

							 ***Table 4*** 

*Average Annual Expenditures on Motor Vehicles per Household by Income and Number of Earners (2012-2013)		
table income_cat noearner_cat if ref_yr >=2012, c(mean motorvehcost_adj) 

log close
translate " G:\Interns\Fall 2014 Interns\Tatenda K Mabikacheche\OPSC\OPSC\Data\Intermediate\Section1_4table_edit.smcl" "G:\Interns\Fall 2014 Interns\Tatenda K Mabikacheche\OPSC\OPSC\Data\Intermediate\Section1.4table_e.pdf"

***************************************************************************************************************************************************************************************************************************************************************************************

/*Consumption Calibration Guide 1 : Clean up and Tabulation

							*******************************
							** STEP 4: Tabulate the Data **
							*******************************

										***1***
* Find the average dollar amount spent on each consumption category by calendar year.Plot these trends.*

use "${path}/Intermediate/categorizedcex.dta", clear
collapse (sum) cost_wcpi [aw= finlwt21] , by (ref_yr ref_mo category)
br
collapse (mean) cost_wcpi , by (ref_yr category)
scatter cost_wcpi ref_yr || lfit cost_wcpi ref_yr, by(category)
scatter cost_wcpi ref_yr || lfit cost_wcpi ref_yr,  by(category, yrescale xrescale ixtitle)
*scatter cost_wcpi ref_yr || lfit cost_wcpi ref_yr, ytitle("Consumption Cost Adjusted for Inflation(2013 Dollars)") ytitle(, size(small)) ylabel(, angle(horizontal))  by(category, yrescale iylabel ixlabel iytitle ixtitle)
graph export "${path}\Intermediate\Graphs\Figure1_AvgConsumptionYr.pdf

save "${path}/Intermediate/collapse/collapse_cost_yr_cat.dta", replace


										***2***
*Find the fraction of income spent on each consumption category by calendar year.Plot these trends.*									
use "${path}/Intermediate/categorizedcex.dta", clear
collapse (sum) value_incwcpi [aw= finlwt21] , by(ref_yr ref_mo)
collapse (mean) value_incwcpi, by(ref_yr)
drop if value_incwcpi ==0
br
scatter value_incwcpi ref_yr|| lfit value_incwcpi ref_yr , ytitle("Income Value Adjusted for Inflation (2013 Dollars)") ytitle(, size(small)) ylabel(, angle(horizontal)) xtitle("Reference Calendar Year") xtitle(, size(small)) title("Average Monthly Income of Population per Calendar Year") title(, size(medium)) 
graph save Graph "E:\datasets\Tatenda\OPSC\Data\Intermediate\Graphs\Average Monthly Income for Population per Calendar year.gph", replace
graph export "${path}\Intermediate\Graphs\Figure2a_AvgIncomeYr.pdf", replace
 
save "${path}/Intermediate/collapse/collapse_income_moyr.dta", replace

merge 1:m ref_yr using "${path}/Intermediate/collapse/collapse_cost_yr_cat.dta"
gen avgexpperinc = cost_wcpi/value_incwcpi
br
****
drop if cost_wcpi ==0
scatter avgexpperinc ref_yr || lfit avgexpperinc ref_yr , by(category, yrescale ixtitle)
scatter avgexpperinc ref_yr || lfit avgexpperinc ref_yr , ytitle("Fraction of Income Spent on Consumption") ytitle(, size(small)) ylabel(, angle(horizontal)) by(category, yrescale ixtitle)
*need to figure out how to add title of whole graph and reorganize plot area with code not manually
graph save Graph "E:\datasets\Tatenda\OPSC\Data\Intermediate\Graphs\Fraction of Monthly Income Spent on Consumption by Population per Calendar Year.gph", replace
graph export "${path}\Intermediate\Graphs\Figure2b_FractionIncomeSpentYr.pdf", replace
save "${path}/Intermediate/collapse/collapse_comboinccost_yr_cat.dta", replace
 

										***3***
* Find the average dollar amount spent on each consumption category by age of head of household. Plot these life-cycle profiles.* 
use "${path}/Intermediate/categorizedcex.dta", clear
collapse (sum) cost_wcpi [aw= finlwt21], by (age_ref ref_mo category)
br
***
drop if cost_wcpi ==0
collapse (mean) cost_wcpi, by (age_ref  category)
scatter cost_wcpi age_ref || lfit cost_wcpi age_ref , by(category)
scatter cost_wcpi age_ref || lfit cost_wcpi age_ref , ytitle("Consumption Cost Adjusted for Inflation(2013 Dollars)") ytitle(, size(small)) ylabel(, angle(horizontal)) by(category, yrescale ixtitle)
*need to figure out how to add title of whole graph and reorganize plot area with code not manually
graph save Graph "E:\datasets\Tatenda\OPSC\Data\Intermediate\Graphs\Average Monthly Consumption by Age.gph"
graph export "${path}\Intermediate\Graphs\Figure3_AvgConsumptionAge.pdf", replace
save "${path}/Intermediate/collapse/collapse_cost_moage_cat.dta", replace


										***4***
* Find the fraction of income spent on each consumption category by age of head of household. Plot these life-cycle profiles. *									
use "${path}/Intermediate/categorizedcex.dta", clear
collapse (sum) value_incwcpi [aw= finlwt21], by(age_ref ref_mo)
br
collapse (mean) value_incwcpi, by(age_ref)
scatter value_incwcpi age_ref|| lfit value_incwcpi age_ref , ytitle("Income Value Adjusted for Inflation (2013 Dollars)") ytitle(, size(small)) ylabel(, angle(horizontal)) xtitle("Reference Calendar Year") xtitle(, size(small)) title("Average Monthly Income of Population by Age") title(, size(medium)) 
graph save Graph "E:\datasets\Tatenda\OPSC\Data\Intermediate\Graphs\Average Monthly Income for Population by Age.gph", replace
graph export "${path}\Intermediate\Graphs\Figure4a_AvgIncomeAge.pdf
save "${path}/Intermediate/collapse/collapse_income_agemo.dta", replace

merge 1:m age_ref using "${path}/Intermediate/collapse/collapse_cost_moage_cat.dta"
gen avgincagecon = cost_wcpi/value_incwcpi
scatter avgincagecon age_ref || lfit avgincagecon age_ref , ytitle("Fraction of Income Spent on Consumption") ytitle(, size(small)) ylabel(, angle(horizontal)) by(category, yrescale ixtitle)
*need to figure out how to add title of whole graph and reorganize plot area with code not manually
graph save Graph "E:\datasets\Tatenda\OPSC\Data\Intermediate\Graphs\Fraction of Monthly Income Spent on Consumption by Age.gph", replace
graph export "${path}\Intermediate\Graphs\Figure4b_FractionIncomeSpentAge.pdf",replace
save "${path}/Intermediate/collapse/collapse_comboinccost_age_cat.dta" , replace


											***5***
* Find the average dollar amount spent on each consumption category by income percentile (so create groups of people for each percentile of household income). Plot how consumption varies by income.*					
use "${path}/Intermediate/categorizedcex.dta", clear
collapse (sum) cost_wcpi [aw= finlwt21] , by(incpct ref_mo category)
collapse (mean) cost_wcpi , by(incpct category)
br
***
label var incpct "10 quantiles of Income"
drop if cost_wcpi ==0
scatter cost_wcpi incpct  || lfit cost_wcpi incpct  , ytitle("Consumption Cost Adjusted for Inflation (2013 Dollars)") ytitle(, size(medsmall)) ylabel(, angle(horizontal)) by(category, yrescale ixtitle)
*need to figure out how to add title of whole graph and reorganize plot area with code not manually
*try putting title after scatter before lfit
*need to rename income percentile ("10 quantiles of Income")
 graph save Graph "E:\datasets\Tatenda\OPSC\Data\Intermediate\Graphs\Average Monthly Consumption by Income Percentile.gph",replace
graph export "${path}\Intermediate\Graphs\Figure5_AvgConsumptionIncpct.pdf", replace
save "${path}/Intermediate/collapse/collapse_cost_incpct_cat.dta", replace

************************************************************
											***6***
* Find the fraction of income spent on each consumption category by income percentile(so create groups of people for each percentile of household income). Plot how consumption varies by income.
use "${path}/Intermediate/categorizedcex.dta",clear
collapse (sum)value_incwcpi [aw= finlwt21] , by(incpct ref_mo)
br
collapse (mean)value_incwcpi , by(incpct)
drop if value_incwcpi ==0
label var incpct "10 quantiles of Income"
scatter value_incwcpi incpct
scatter value_incwcpi incpct|| lfit value_incwcpi incpct , ytitle("Income Value Adjusted for Inflation (2013 Dollars)") ytitle(, size(small)) ylabel(, angle(horizontal)) xtitle("10 quantiles of Income") xtitle(, size(small)) title("Average Monthly Income of Population by Income Percentile") title(, size(medium)) 
graph save Graph "E:\datasets\Tatenda\OPSC\Data\Intermediate\Graphs\Average Monthly Income for Population by Income Percentile.gph", replace
graph export "${path}\Intermediate\Graphs\Figure6a_AvgIncomeIncpct.pdf",replace
save "${path}/Intermediate/collapse/collapse_income_incpctmo.dta", replace

merge 1:m incpct using "${path}/Intermediate/collapse/collapse_cost_incpct_cat.dta"
gen fracconbyincpct = cost_wcpi/value_incwcpi
scatter fracconbyincpct incpct, by(category)
scatter fracconbyincpct incpct || lfit fracconbyincpct incpct , ytitle("Fraction of Income Spent on Consumption") ytitle(, size(small)) ylabel(, angle(horizontal)) by(category, yrescale ixtitle)
graph save Graph "E:\datasets\Tatenda\OPSC\Data\Intermediate\Graphs\Fraction of Monthly Income Spent on Consumption by Income Percentile.gph"
graph export "${path}\Intermediate\Graphs\Figure6b_FractionIncomeSpentIncpct.pdf
save "${path}/Intermediate/collapse/collapse_comboinccost_incpct_cat.dta", replace

forvalues x =1/17{
 scatter fracconbyincpct incpct if category ==`x' 
} 

***Interesting  graphs
scatter fracconbyincpct category if  incpct ==1
scatter fracconbyincpct category if  incpct ==5
scatter fracconbyincpct category if  incpct ==10

 /*twoway (scatter value_incwcpi ref_yr), ytitle("Income value adjusted for inflation") ytitle(, size(small)) yscale(range(500000 1500000)) xtitle(Reference Calendar Year) xtitle(, size(small)) title(Average Monthly Income of Population per Calendar Year)
`"Income Value Adjusted for Inflation (2013 dollars)"' */

*/
						

