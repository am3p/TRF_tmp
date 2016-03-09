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

__declspec(dllexport) struct VBAResult __stdcall Pricer_MC(float FXPrice, long FXSize,
														   long YTMType, long YTMTNum, float* YTMT, float* YTM,
														   long* RatedType, long* RatedTNum, float* RatedT, float* Rated,
														   long* RatefType, long* RatefTNum, float* RatefT, float* Ratef,
														   long* VolType, long* VolTNum, long* VolKNum, float* VolT, float* VolK, float* Vol,
														   long NSchedule, long* T_exp, long* T_pay, long* BermudanType, long* PayoffType, long* RefPriceType,
														   float* Strike, float AccRet,
														   long Mode, long SimN, long blockN, long threadN, long isStrikePriceQuote){

	long i, j, k; float s;

	// Stock: Current stock prices
	float FXPrice_[FXSizeMax] = {0};
	// Temporary variables for a number of underlying
	long FXSize_ = FXSize;
	
	// YTM rate: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	long YTMType_ = 0, YTMSize_ = 0;
	float YTMt_[RateTMax] = {0}, YTM_[RateTMax] = {0};
	long YTMInd_src = 0, YTMInd_dest = 0;
	
	// Domestic risk-free rate info: type and size (type: 0 - Fixed / 1 - Term)
	long RatedType_[FXSizeMax] = {0}, RatedSize_[FXSizeMax] = {0};
	// Rate info: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	float Ratedt_[FXSizeMax * RateTMax] = {0}, Rated_[FXSizeMax * RateTMax] = {0};
	long RatedInd_src = 0, RatedInd_dest = 0;

	// Foreign risk-free rate info: type and size (type: 0 - Fixed / 1 - Term)
	long RatefType_[FXSizeMax] = {0}, RatefSize_[FXSizeMax] = {0};
	// Div info: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	float Rateft_[FXSizeMax * RateTMax] = {0}, Ratef_[FXSizeMax * RateTMax] = {0};
	long RatefInd_src = 0, RatefInd_dest = 0;

	// Vol info: type and size (type: 0 - Fixed / 1 - Term / 2 - Surface)
	long VolType_[FXSizeMax] = {0}, VolSize_t_[FXSizeMax] = {0}, VolSize_K_[FXSizeMax] = {0};
	// Vol info: fixed case, term structure, and surface
	// Time axis size: Max 40 (Covering NICE Full data)
	// Price axis size for vol surface: Max 21 (Covering NICE Full data)
	float Volt_[FXSizeMax * VolTMax] = {0}, VolK_[FXSizeMax * VolKMax] = {0}, Vol_[FXSizeMax * VolTMax * VolKMax] = {0};
	long VolTInd_src = 0, VolTInd_dest = 0, VolKInd_src = 0, VolKInd_dest = 0, VolInd_src = 0, VolInd_dest = 0;

	// Correlation: Cholesky decomposed (LD), Max 4x4 (just my purpose)
	float FXCorr_LD[FXSizeMax * FXSizeMax] = {0};

	// Quanto adjustment
	float QuantoAdj[FXSizeMax] = {0};
	long QuantoInd_src = 0, QuantoInd_dest = 0;

	// Schedule info: TTM
	long PayoffT_[ScheduleSizeMax] = {0}, PayoffT_pay[ScheduleSizeMax] = {0};
	// Schedule info: Types (Bermudan / Payoff type / Reference price type)
	long BermudanType_[ScheduleSizeMax] = {0}, PayoffType_[ScheduleSizeMax] = {0}, RefPriceType_[ScheduleSizeMax] = {0};
	// Schedule info: Exercise
	float PayoffK_[ScheduleSizeMax] = {0};
	float Accret_ = 0;
	

	// Result format
	struct VBAResult* result = (struct VBAResult *) malloc(sizeof(struct VBAResult));
	struct VBAResult result_VBA;
	result->price = 0;
	//result->theta = 0;
	for (i = 0; i < FXSize_; i++){
		result->delta[i] = 0;
		result->vega[i] = 0;
		result->rho[i] = 0;
		result->rhod[i] = 0;
		result->rhof[i] = 0;
	}
	for (i = 0; i < 100; i++){
		result->prob[i] = 0;
	}

	// Copying product info for CUDA function
	for (i = 0; i < FXSize_; i++){
		// Current stock price (normalized by base price)
		FXPrice_[i] = FXPrice;
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
	for (i = 0; i < FXSize_; i++){
		RatedType_[i] = RatedType[i];
		RatedSize_[i] = RatedTNum[i];

		RatefType_[i] = RatefType[i];
		RatefType_[i] = RatefTNum[i];

		switch(RatedType_[i]){
			case 0:
				{
					Rated_[RatedInd_dest] = Rated[RatedInd_src];
					RatedInd_src++;
					RatedInd_dest = (i+1)*RateTMax;
					break;
				}
			case 1:
				{
					for (j = 0; j < RatedSize_[i]; j++){
						Ratedt_[RatedInd_dest] = RatedT[RatedInd_src];
						Rated_[RatedInd_dest] = Rated[RatedInd_src];
						RatedInd_src++;
						RatedInd_dest++;
					}
					RatedInd_dest = (i+1)*RateTMax;
					break;
				}
			default:
				break;
		}

		switch(RatefType_[i]){
			case 0:
				{
					Ratef_[RatefInd_dest] = Ratef[RatefInd_src];
					RatefInd_src++;
					RatefInd_dest = (i+1)*RateTMax;
					break;
				}
			case 1:
				{
					for (j = 0; j < RatefSize_[i]; j++){
						Rateft_[RatefInd_dest] = RatefT[RatefInd_src];
						Ratef_[RatefInd_dest] = Ratef[RatefInd_src];
						RatefInd_src++;
						RatefInd_dest++;
					}
					RatefInd_dest = (i+1)*RateTMax;
					break;
				}
			default:
				break;
		}
	}

	// Vol info
	for (i = 0; i < FXSize_; i++){
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
						VolInd_src++; VolInd_dest++;
						VolTInd_src++; VolTInd_dest++;
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

					for (j = 0; j < VolKNum[i]; j++){
						VolK_[VolKInd_dest] = VolK[VolKInd_src];
						VolKInd_src++; VolKInd_dest++;
					}
					VolKInd_dest = (i+1)*VolKMax;

					for (j = 0; j < VolTNum[i]; j++){
						for (k = 0; k < VolKNum[i]; k++){
							Vol_[VolInd_dest] = Vol[VolInd_src];
							VolInd_src++; VolInd_dest++;
						}
						VolInd_dest = i*VolTMax*VolKMax + (j+1)*VolKMax;
					}
					VolInd_dest = (i+1)*VolTMax*VolKMax;
					break;
				}
			default:
				break;
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
	}
	Accret_ = AccRet;

	// Modify!!
	// MC function
	CalcMC(FXPrice_, FXSize_,
		   NSchedule,	
		   PayoffT_, PayoffT_pay, PayoffType_, RefPriceType_,
		   PayoffK_, Accret_,
		   RatedType_, RatedSize_, Ratedt_, Rated_,
		   RatefType_, RatefSize_, Rateft_, Ratef_,
		   VolType_, VolSize_t_, VolSize_K_, Volt_, VolK_, Vol_,
		   YTMType_, YTMSize_, YTMt_, YTM_,
		   isStrikePriceQuote, SimN, Mode, blockN, threadN,
		   result);

	// Arrange result
	result_VBA.price = result->price;
	//for (i = 0; i < ScheduleSizeMax; i++){
	//	result_VBA.prob[i] = result->prob[i];
	//}
	//if (Mode > 0){
	//	for (i = 0; i < NStock; i++)
	//		result_VBA.delta[i] = result->delta[i];
	//	for (i = 0; i < NStock; i++)
	//		result_VBA.gamma[i] = result->gamma[i];
	//	for (i = 0; i < NStock; i++)
	//		result_VBA.vega[i] = result->vega[i];
	//}
	//if (Mode > 1){
	//	for (i = 0; i < NStock; i++)
	//		result_VBA.rho[i] = result->rho[i];
	//	result_VBA.theta = result->theta;
	//}
	//if (Mode > 2){
	//	for (i = 0; i < NStock; i++)
	//		result_VBA.vanna[i] = result->vanna[i];
	//	for (i = 0; i < NStock; i++)
	//		result_VBA.volga[i] = result->volga[i];
	//}

	free(result);
	return result_VBA;
}



