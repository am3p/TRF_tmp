#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

// MC function
void CalcMC(float* FXPrice_, long FXSize_,
			long ScheduleSize_,	
			long* PayoffT_, long* PayoffT_pay, long* PayoffType_, long* RefPriceType_,
			float* PayoffK_, float Accret_,
 			long* RatedType_, long* RatedSize_, float* Ratedt_, float* Rated_,
			long* RatefType_, long* RatefSize_, float* Rateft_, float* Ratef_,
 			long* VolType_, long* VolSizet_, long* VolSizeK_, float* Volt_, float* VolK_, float* Vol_,
			long YTMType_, long YTMSize_, float* YTMt_, float* YTM_,
			long isStrikePriceQuote_, long SimN_, long SimMode_, long blockN_, long threadN_,
			struct VBAResult* result);


#ifdef __cplusplus
}
#endif