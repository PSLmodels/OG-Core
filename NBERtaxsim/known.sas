options source2;

if error ge 1 then skip = -1; else skip = 0;

if error gt 5 then do;
/* SSBEN in AGI neglected, but no excuse */
if abs(c02500-e02500) gt 25 and SSIND EQ 0 then do;
  skip = 101;
  put FLPDYR SOIYR RECID 'ssben in agi neglected, but no excuse given in SSIND';
end;

if abs(c04800-e04800) lt 5 and abs(c05200-e05200) gt 5 then do;
  skip = 102;
  put FLPDYR SOIYR RECID 'XYZ tax tables not read correctly';
end;

if e87480 gt 1200 and e87480 eq e87481 then do;
   skip = 103;
   put FLPDYR SOIYR RECID 'forgot haircut on form 8863';
end;

/* Problem on Schedule D */
if FLPDYR gt 1997 and SCHD and e24517+e24520 lt e04800 then do;
  skip=110;
  put FLPDYR SOIYR RECID ' e2451+e24520 is less than e04800 on schedule D';
end;

/* Report zero for e23650 and e23250 but e24510 is 0 */
if e23650 eq 0 and e23250 eq 0 and e24510 gt 0 then do;
  skip = 111;
  put FLPDYR SOIYR RECID ' Reports zero for e23650 and e23250 and e24510 on Schedule D';
end;

if SCHD and flpdyr ge 1997 and abs(e24520-max(0,e04800-e24517)) gt 25 then do;
  skip = 113;
  put FLPDYR SOIYR RECID ' taxpayers did not fill out schD worksheet correctly';
end;

/* SchdD taxpayer error not corrected by SOI */
if SCHD and abs(e24510-min(e23250,e23650)) gt 5 and 
            min(e23250,e23650) gt 0 then do; 
  skip=114;
  put FLPDYR SOIYR RECID ' inconsistent e24510 e23250 e23650';
end;

/* SOI doesn't recognize all the children */
if abs(c05800-e05800) LT 25 and E07220 LT C07220 then do;
 *  put FLPDYR SOIYR RECID ' SOI misses some children for ctc';
end;

/* No Schedule A local tax, but added to ALMINY on 6251 */
if (e18300-e18600) eq 0 and e60240 GT 0 then do; 
   skip = 119;
   put FLPDYR SOIYR RECID ' No local tax on Sch A, but included on 6251';
end;

/* TXST is 10 but taxpayer used e05200 or e24580 for tax */
if f2555 gt 0 and 
   abs(e05800-e05200) lt 5 then do; 
   skip=121;
   put FLPDYR SOIYR RECID ' TXST is 10 but taxpayer used e05200 or e24580 for tax';
end;
 
if f2555 gt 0 and e24580 lt e05800 and abs(e24580-e05800) lt 5 then do;
   skip=122;
   put FLPDYR SOIYR RECID ' taxpayer ignored f2555 when calculating tax';
end;

/* Mysterious other taxes */
if abs(e05700-c05700) gt 5 then do;
   skip=123;
   put FLPDYR SOIYR RECID ' myterious other taxes';
end;

/* Foreign income exclusion but no Form 2555 */
if f2555 eq 0 and E02700 gt 0 then do;
  skip = 124;
  put FLPDYR SOIYR RECID ' Foreign income exclusion but no form 2555';
end;

/* feided but no f2555 */
if e02700 GT 0 and (TXST NE 10 or F2555 EQ 0) then do;
   skip = 125;
   put FLPDYR SOIYR RECID ' _feided but tax status not 10';
end;

/* Rate Reduction Credit taken on 1040 */
if FLPDYR eq 2001 and c07970 LT e07970 then do;
   skip = 126;
end;

/* Tax on Form 6251 not carried over to 1040 */
if f6251 eq 1 and e63200+25 LT e09600 then do;
   skip = 127; 
   put FLPDYR SOIYR RECID ' Tax on Form 6251 is not carried over to 1040';
end;

/* Taxpayer didn't include e02700 in Alternative Minimum Income */
if flpdyr ge 2006 and f6251 eq 1 and abs(_alminc-(c62700+e02700)) gt 5 then do; 
  skip = 128;
  put FLPDYR SOIYR RECID ' Taxpayer did not include e02700 in alternative minimum income';
end;

/* Taxpayer doesn't take full standard deduction for unknown reason */
if e04100+25 LT c04100 and dsi eq 0 and e04470 eq 0 and e04600 NE 0 and e04800 gt 0 then do; 
   skip = 129;
   put FLPDYR SOIYR RECID ' Taxpayer does not take full standard deduction for unknown reason';
end;

/* Vehicle tax  */
if FLPDYR in(2009:2010) and F6251 eq 1 and abs(e60240-(e18300-e18600)) gt 25 then do; 
   skip = 130;
   put FLPDYR SOIYR RECID ' Problem with vehicle tax';
end;

/* Charity */
if abs(e19700-c19700) gt 25 and e20200 gt 25 then do;
   skip = 131;
   put FLPDYR SOIYR RECID ' Charity carryover';
end;

/* SOI recodes Schedule A to zeroes */
if e04470 eq 0 and t04470  GT 0 and f6251 eq 1 and abs(e60000-t04470) LT 5 then do; 
   skip=132;
   put FLPDYR SOIYR RECID ' put SOI recoded Sched A to zeroes and problems remain on 6251';
end;

/* Standard deduction not used in AMT calculation */
if t04470 GT 0 and e04470 EQ 0 and c04470 EQ 0 and abs(e04800-c04800-c04100) LT 5 then do; 
   skip=133;
   put FLPDYR SOIYR RECID ' Standard deduction not used in AMT calculation';
end;

/* Standard deduction not taken on AMT when allowed */
if flpdyr LT 2002 & _standard GT 0 and _standard eq abs(c62100-e62100) then do; 
   skip = 134;
   put FLPDYR SOIYR RECID ' Standard deduction not taken on AMT when allowed';
end;

/* AMT gt 0 but f6251 eq 0 */
if e09600 gt 0 and f6251 eq 0 then do; 
   skip = 135;
   put FLPDYR SOIYR RECID ' AMT (e09600) gt 0 but f6251 eq 0' e09600= f6251=;
end;

/* AMT should be paid, but no 6251 */
if e09600 eq 0 and c09600 gt 0 then do;
   skip = 136;
   put FLPDYR SOIYR RECID ' AMT (c09600) gt 0 no f6251' c09600=;
end;

/* General Business Credit is zeroed out by editor in E07400 but still included in total E07100 */
if t07400 gt 0 and e07400 eq 0 and (e07100-c07100) eq t07400 then do;
   skip = 143; 
   put FLPDYR SOIYR RECID ' Gen Bus Credit zeroed by editor in e07400 but still included in total e07100';
end;

/* Alternative fueled vehicle credit */
if e08001 eq e08000 and e08000 gt 0 and abs(c07100-e07100-e08000) lt 25 then do;
   skip = 144;
   put FLPDYR SOIYR RECID ' Alternative Fueled vehicle credit missed';
end;

/* e07400 edited from t07400 but not zeroed */
if e07400 gt 0 and e07400 ne t07400 then do;
   skip = 146;
   put FLPDYR SOIYR RECID ' e07400 edited from t07400 but not zeroed';
end;

/* e09400 subtracted e09200 but not from e10300 */
if abs(e10300-e10300-e09400) lt 5 and e09400 ne 0 then do;
   skip=147;
   put FLPDYR SOIYR RECID ' e09400 subtracted from e09200 but not from e10300';
end;

if error ne 0 and abs(error-e09400) lt 5 then do;
  skip = 151;
  put FLPDYR SOIYR ' e09400 not accounted for ' e01300= e09200= e09400=;
end;

/*
if f8863 gt 0 and min(e87483,1)+min(e87488,1)+min(e87493,1)+min(e87498,1) ne n30 then do;
  skip = 152;
  put FLPDYR SOIYR RECID ' Wrong number of students for AOC ' e87483= e87488= e87493= e87498= n30=;
end;
if flpdyr eq 2011 and recid eq 7920 then put recid= e87483= e87488= e87493= e87498= n30=;
*/

if f2555*F6251 gt 0 and e09600 ne s09600 then do;
  skip = 153;
  put FLPDYR SOIYR RECID ' Form 2555 incorrect edit ' ;
end;

if f2555 gt 0 and _oldfei eq e05100 then do;
   skip = 154;
   put FLPDYR SOIYR RECID '  Taxpayer took FEIE rather than credit' ;
end;

if flpdyr eq 2011 and f6251 eq 1 and f2555 gt 0 and error gt 5 then do;
   skip = 155;
   put FLPDYR SOIYR RECID ' E02700 neglected on AMT F2555 worksheet';
end;

if FLPDYR NOT in(1993:2011) then do;
   skip = 1992;
end;

*if FLPDYR ne SOIYR then skip = 299;
if FLPDMO ne 12 then skip = 202;
if TXST eq 9 or SCHJIN or SCHJ then skip = 203;

/* We do not observe Advance Payment of Rate Reduction Credit */
if abs(e04800-c04800) lt 5 then do;
   errr=abs(c08795-e08795);
   if flpdyr eq 2001 and (   
   (mars in(1,3,6) and errr lt 305) or     
   (mars in(4,7)   and errr lt 505) or
   (mars eq 2      and errr lt 605)) then skip = 204;
end;

/* Advance payment of CTC not observed*/
diff = int(abs(e07220-c07220)+.499);
diff = 10*int((diff+5)/10);
if FLPDYR eq 2003 and diff in(400,500,600,800,1000,1200,1500,1600,1800,2000) then skip=205;
if FLPDYR eq 1999 and diff in(500,1000,1500,2000) then skip=205;
if FLPDYR eq 2003 and _taxbc LT _precrd then skip=205;


/* Taxpayer didn't use F2555 worksheet and wasn't caught */
if e02700 GT 0 and e05100 eq e05800 then skip = 206;
end;

/* SOI doesn't recognize all the children */
if abs(c05800-e05800) LT 25 and E07220+25 LT C07220 then do;
   skip=115;
   put FLPDYR SOIYR RECID ' SOI misses some children for ctc';
end;

