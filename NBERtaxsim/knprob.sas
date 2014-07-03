options pagesize=max ls=80;
options nocenter;
libname t "/research/indcon/drf";
%INCLUDE "./taxcalc.sas";
data returns;
%include "filelist.sas";
*set t.subset;
*set t.tenth;
*set t.supset;
*set t.mix;

if FLPDYR ge 1993;
_exact = 0;

%INIT;
%COMP;
smooth=c10300/1e6;

_exact = 1;
%COMP;

error = abs(c10300-e10300);

file 'probrecs.txt';
%include "known.sas";
if error lt 5 then error = 0;
if error lt 5 then iserr=0; else iserr=1;
if error gt 0 and skip lt 200 then realerr=1;else realerr=0;
weight = s006/100;
error  = error/1e6;
c10300 = c10300/1e6;
e10300 = e10300/1e6;
keep flpdyr soiyr smooth c10300 e10300 skip error iserr realerr s006 c10300 e10300 weight;
run;

proc means n min max mean sum;
weight weight;
run;

title "weighted sum of absolute errors in millions of dollars";
proc tabulate noseps;
class flpdyr skip;
var error iserr realerr c10300 e10300;
weight weight;
table skip all,n*f=8.0 error*sum*f=8.2;
table flpdyr all,n*f=8.0 iserr*sum*f=8.0 error*sum*f=8.2;
table flpdyr all,(all skip)*error*sum*f=8.2;
table flpdyr all,(e10300 c10300 realerr iserr)*sum*f=10.0 (all skip)*N*f=8.0;
run;


proc reg;
title "Weighted regression";
weight s006;
model c10300 = e10300;
model smooth = e10300;
run;

proc reg;
title "Unweighted regression";
model c10300 = e10300;
model smooth = e10300;
run;

proc reg;
title "Exclude some records, unweighted";
where skip lt 200 and flpdyr ne 2001 and flpdyr ne 2003;
model c10300  = e10300;
model smooth = e10300;
run;

