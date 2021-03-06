///////////////////////////////////////////////////////////////////////////////////////
//
// MC Pricer ver 0.1
// 2014. 11. 28 by Boram Hwang
//
// Main features
//  1. Conducting MC pricing for user-defined payoff
//     (This version only contains knock-out call, and composites of step-down ELS,
//		but payoffs are quite easily extendable)
//  2. MC is done by GPU-wise sample construction
//     (Each sample is constructed per a GPU so that by upgrading VGA with more SM
//		will reduce calculation time)
//  3. Adapting constant, term-structured, or surface parameters
//	   (longerpolation/Extrapolation of parameters can be done linearly only, in this
//		version)
//
///////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

#include "MCwrapper_fcn.h"
#include "MCstruct_VBA.h"
#include "VariableSize.h"

__declspec(dllexport) struct VBAResult __stdcall Pricer_MC(long NStock, double* StockPrice, double* BasePrice,
														   long YTMType, long YTMTNum, double* YTMT, double* YTM,
														   long* DRateType, long* DRateTNum, double* DRateT, double* DRate,
														   long* FRateType, long* FRateTNum, double* FRateT, double* FRate,
														   long* VolType, long* VolTNum, long* VolKNum, double* VolT, double* VolK, double* Vol,
														   long* FXVolType, long* FXVolTNum, double* FXVolT, double* FXVol,
														   double* StockCorr,  double* FXCorr,
														   long NSchedule, long* T_exp, long* T_pay, long* BermudanType, long* PayoffType, long* RefPriceType,
														   double* Strike, double UpAmt, double DownAmt, double AccRet_KO, double AccRet, double* Participation,
														   long Mode, long SimN, long blockN, long threadN, 
														   long isStrikePriceQuote, long VolInterpMode){

	long i, j, k; double s;

	// Stock: Current stock prices
	double StockPrice_[StockSizeMax] = {0};
	// Stock: Base prices for the product
	double BasePrice_[StockSizeMax] = {0};

	
	// YTM rate: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	long YTMType_ = 0, YTMSize_ = 0;
	double YTMt_[RateTMax] = {0}, YTM_[RateTMax] = {0};
	long YTMInd_src = 0, YTMInd_dest = 0;
	
	// Rate info: type and size (type: 0 - Fixed / 1 - Term)
	long DRateType_[StockSizeMax] = {0}, DRateSize_[StockSizeMax] = {0};
	// Rate info: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	double DRatet_[StockSizeMax * RateTMax] = {0}, DRate_[StockSizeMax * RateTMax] = {0};
	long DRateInd_src = 0, DRateInd_dest = 0;

	// Rate info: type and size (type: 0 - Fixed / 1 - Term)
	long FRateType_[StockSizeMax] = {0}, FRateSize_[StockSizeMax] = {0};
	// Rate info: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	double FRatet_[StockSizeMax * RateTMax] = {0}, FRate_[StockSizeMax * RateTMax] = {0};
	long FRateInd_src = 0, FRateInd_dest = 0;

	// Vol info: type and size (type: 0 - Fixed / 1 - Term / 2 - Surface)
	long VolType_[StockSizeMax] = {0}, VolSize_t_[StockSizeMax] = {0}, VolSize_K_[StockSizeMax] = {0};
	// Vol info: fixed case, term structure, and surface
	// Time axis size: Max 40 (Covering NICE Full data)
	// Price axis size for vol surface: Max 21 (Covering NICE Full data)
	double Volt_[StockSizeMax * VolTMax] = {0}, VolK_[StockSizeMax * VolTMax * VolKMax] = {0}, Vol_[StockSizeMax * VolTMax * VolKMax] = {0};
	long VolTInd_src = 0, VolTInd_dest = 0, VolKInd_src = 0, VolKInd_dest = 0, VolInd_src = 0, VolInd_dest = 0;

	// Correlation: Cholesky decomposed (LD), Max 4x4 (just my purpose)
	double StockCorr_LD[StockSizeMax * StockSizeMax] = {0};

	// Quanto adjustment
	double QuantoAdj[StockSizeMax] = {0};
	long QuantoInd_src = 0, QuantoInd_dest = 0;

	// Schedule info: TTM
	long PayoffT_[ScheduleSizeMax] = {0}, PayoffT_pay[ScheduleSizeMax] = {0};
	// Schedule info: Types (Bermudan / Payoff type / Reference price type)
	long BermudanType_[ScheduleSizeMax] = {0}, PayoffType_[ScheduleSizeMax] = {0}, RefPriceType_[ScheduleSizeMax] = {0};
	// Schedule info: Exercise and barrier
	double PayoffK_[ScheduleSizeMax] = {0}, UpBarrier_[ScheduleSizeMax] = {0}, DownBarrier_[ScheduleSizeMax] = {0};
	// Schedule info: Payout coupon, dummy
	double Coupon_[ScheduleSizeMax] = {0}, Dummy_[ScheduleSizeMax] = {0};
	// Schedule info: Participation rate
	double Participation_[ScheduleSizeMax] = {0};

	// Result format
	struct VBAResult* result = (struct VBAResult *) malloc(sizeof(struct VBAResult));
	struct VBAResult result_VBA;
	result->price = 0;
	result->theta = 0;
	for (i = 0; i < NStock; i++){
		result->delta[i] = 0;
		result->gamma[i] = 0;
		result->vega[i] = 0;
		result->rho[i] = 0;
		result->vanna[i] = 0;
		result->volga[i] = 0;
	}
	for (i = 0; i < 100; i++){
		result->prob[i] = 0;
	}
	result->coupon = 0;

	// Copying product info for CUDA function
	for (i = 0; i < NStock; i++){
		// Current stock price (normalized by base price)
		StockPrice_[i] = StockPrice[i];
		BasePrice_[i] = BasePrice[i];
	}

	// YTM info
	YTMType_ = YTMType;
	YTMSize_ = YTMTNum;
	switch(YTMType_){
		case 0:
			{
				YTM_[YTMInd_dest] = YTM[YTMInd_src];
				YTMInd_src++;
				YTMInd_dest = RateTMax;
				break;
			}
		case 1:
			{
				for (i = 0; i < YTMSize_; i++){
					YTMt_[YTMInd_dest] = YTMT[YTMInd_src];
					YTM_[YTMInd_dest] = YTM[YTMInd_src];
					YTMInd_src++;
					YTMInd_dest++;
				}
				YTMInd_dest = RateTMax;
				break;
			}
		default:
			break;
	}

	// Risk free rate info
	DRateInd_src = DRateInd_src;
	for (i = 0; i < NStock; i++){
		DRateType_[i] = DRateType[i];
		DRateSize_[i] = DRateTNum[i];
		switch(DRateType_[i]){
			case 0:
				{
					DRate_[DRateInd_dest] = DRate[DRateInd_src];
					DRateInd_src++;
					DRateInd_dest = (i+1)*RateTMax;
					break;
				}
			case 1:
				{
					for (j = 0; j < DRateSize_[i]; j++){
						DRatet_[DRateInd_dest] = DRateT[DRateInd_src];
						DRate_[DRateInd_dest] = DRate[DRateInd_src];
						DRateInd_src++;
						DRateInd_dest++;
					}
					DRateInd_dest = (i+1)*RateTMax;
					break;
				}
			default:
				break;
		}
	}

	// Risk free rate info
	FRateInd_src = FRateInd_src;
	for (i = 0; i < NStock; i++){
		FRateType_[i] = FRateType[i];
		FRateSize_[i] = FRateTNum[i];
		switch(FRateType_[i]){
			case 0:
				{
					FRate_[FRateInd_dest] = FRate[FRateInd_src];
					FRateInd_src++;
					FRateInd_dest = (i+1)*RateTMax;
					break;
				}
			case 1:
				{
					for (j = 0; j < FRateSize_[i]; j++){
						FRatet_[FRateInd_dest] = FRateT[FRateInd_src];
						FRate_[FRateInd_dest] = FRate[FRateInd_src];
						FRateInd_src++;
						FRateInd_dest++;
					}
					FRateInd_dest = (i+1)*RateTMax;
					break;
				}
			default:
				break;
		}
	}

	// Vol info
	for (i = 0; i < NStock; i++){
		VolType_[i] = VolType[i];
		VolSize_t_[i] = VolTNum[i];
		VolSize_K_[i] = VolKNum[i];
		switch(VolType_[i]){
			case 0:
				{
					Vol_[VolInd_dest] = Vol[VolInd_src];
					VolInd_src++;
					VolInd_dest = (i+1)*VolTMax*VolKMax;
					break;
				}
			case 1:
				{
					for (j = 0; j < VolTNum[i]; j++){
						Volt_[VolTInd_dest] = VolT[VolTInd_src];
						Vol_[VolInd_dest] = Vol[VolInd_src];						
						VolTInd_src++; VolTInd_dest++;
						VolInd_src++; VolInd_dest++;
					}
					VolTInd_dest = (i+1)*VolTMax;
					VolInd_dest = (i+1)*VolTMax*VolKMax;
					break;
				}
			case 2:
				{
					for (j = 0; j < VolTNum[i]; j++){
						Volt_[VolTInd_dest] = VolT[VolTInd_src];
						VolTInd_src++; VolTInd_dest++;
					}
					VolTInd_dest = (i+1)*VolTMax;

					//for (j = 0; j < VolKNum[i]; j++){
					//	VolK_[VolKInd_dest] = VolK[VolKInd_src];
					//	VolKInd_src++; VolKInd_dest++;
					//}
					//VolKInd_dest = (i+1)*VolKMax;

					for (j = 0; j < VolTNum[i]; j++){
						for (k = 0; k < VolKNum[i]; k++){
							VolK_[VolKInd_dest] = VolK[VolKInd_src];
							Vol_[VolInd_dest] = Vol[VolInd_src];
							VolKInd_src++; VolKInd_dest++;
							VolInd_src++; VolInd_dest++;
						}
						VolKInd_dest = i*VolTMax*VolKMax + (j+1)*VolKMax;
						VolInd_dest = i*VolTMax*VolKMax + (j+1)*VolKMax;
					}
					VolInd_dest = (i+1)*VolTMax*VolKMax;
					break;
				}
			default:
				break;
		}
	}

	// Correlation: Cholesky decomposition (LD)
	for (i = 0; i < NStock; i++){
		for (j = 0; j < (i+1); j++) {
			s = 0;
			for (k = 0; k < j; k++) 
				s += StockCorr[i*NStock+k] * StockCorr[j*NStock+k];
			StockCorr_LD[i*NStock+j] = (i == j) ? sqrt(StockCorr[i*NStock+i]-s) : (1.0/StockCorr_LD[j*NStock+j] * (StockCorr[i*NStock+j]-s));
		}
	}	

	// Quanto adjustment factor
	QuantoInd_src = 0; QuantoInd_dest = 0;
	for (i = 0; i < NStock; i++){
		if (FXVolType[i] == 0){
			QuantoAdj[i] = FXCorr[i] * FXVol[i];
		}
		else if (FXVolType[i] == 1){
			// not yet defined
		}
	}

	// Schedule info: copying relevant information
	for (i = 0; i < NSchedule; i++){
		PayoffT_[i] = T_exp[i];
		PayoffT_pay[i] = T_pay[i];

		BermudanType_[i] = BermudanType[i];
		PayoffType_[i] = PayoffType[i];
		RefPriceType_[i] = RefPriceType[i];

		PayoffK_[i] = Strike[i];

		Participation_[i] = Participation[i];
	}

	// MC function
	CalcMC(NStock, StockPrice_, BasePrice_,
		   NSchedule,	
		   PayoffT_, PayoffT_pay, BermudanType_, PayoffType_, RefPriceType_,
		   PayoffK_, Coupon_, Dummy_,
		   AccRet, AccRet_KO,
		   UpAmt, DownAmt, Participation_,
		   DRateType_, DRateSize_, DRatet_, DRate_,
		   FRateType_, FRateSize_, FRatet_, FRate_,
		   VolType_, VolSize_t_, VolSize_K_, Volt_, VolK_, Vol_,
		   YTMType_, YTMSize_, YTMt_, YTM_,
		   StockCorr_LD, QuantoAdj,
		   isStrikePriceQuote, VolInterpMode, SimN, Mode, blockN, threadN,
		   result);

	// Arrange result
	result_VBA.price = result->price;
	for (i = 0; i < ScheduleSizeMax; i++){
		result_VBA.prob[i] = result->prob[i];
	}
	if (Mode > 0){
		for (i = 0; i < NStock; i++)
			result_VBA.delta[i] = result->delta[i];
		for (i = 0; i < NStock; i++)
			result_VBA.gamma[i] = result->gamma[i];
		for (i = 0; i < NStock; i++)
			result_VBA.vega[i] = result->vega[i];
	}
	if (Mode > 1){
		for (i = 0; i < NStock; i++)
			result_VBA.rho[i] = result->rho[i];
		result_VBA.theta = result->theta;
	}
	if (Mode > 2){
		for (i = 0; i < NStock; i++)
			result_VBA.vanna[i] = result->vanna[i];
		for (i = 0; i < NStock; i++)
			result_VBA.volga[i] = result->volga[i];
	}

	free(result);
	return result_VBA;
}



