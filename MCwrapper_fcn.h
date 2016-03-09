#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

// MC function
void CalcMC(long StockSize_, double* StockPrice_, double* BasePrice_,
			long ScheduleSize_,	
			long* PayoffT_, long* PayoffT_pay, long* BermudanType_, long* PayoffType_, long* RefPriceType_,
			double* PayoffK_, double* Coupon_, double* Dummy_,
			double AccRet, double AccRet_KO,
			double UpAmt, double DownAmt, double* Participation_,
 			long* DRateType_, long* DRateSize_, double* DRatet_, double* DRate_,
			long* FRateType_, long* FRateSize_, double* FRatet_, double* FRate_,
 			long* VolType_, long* VolSizet_, long* VolSizeK_, double* Volt_, double* VolK_, double* Vol_,
			long YTMType_, long YTMSize_, double* YTMt_, double* YTM_,
			double* correl_, double* Quanto_,
			long isStrikePriceQuote_, long VolInterpMode_, 
			long SimN_, long SimMode_, long blockN_, long threadN_,
			struct VBAResult* result);


#ifdef __cplusplus
}
#endif