#pragma once
#include "VariableSize.h"

// See definitions of VBA type definitions in Excel worksheets

struct VBARate{
	long RateType;
	long RateSize;

    float Ratet[RateTMax];
    float Rater[RateTMax];
};

struct VBAVol{
	long VolType;
    long VolSizet;
    long VolSizeK;
    
    float Volt[VolTMax];
    float VolK[VolKMax];
    float Volv[VolTMax * VolKMax];
};

struct VBAUnderlying{
    float S;
	float BasePrice;
    struct VBARate Rfd;
    struct VBARate Rff;
    struct VBAVol Vol;
	float Quanto;
	float Corr_row[FXSizeMax];
};

struct VBAPayoff{
	long T_exp;
	long T_pay;

    long BermudanType;
    long PayoffType;
	long RefPriceType;
    
	float K;
    float UpBarrier;
    float DownBarrier;
    float TotalUpBarrier;
    float TotalDownBarrier;
    float Participation;

    float Coupon;
	float Dummy;
};

struct VBACalcOption{
	long ScheduleNum;
	long SimN;
	long SimMode;
	long blockN;
	long threadN;
	long isStrikePriceQuote;
};

struct VBAResult{
    float price;
	float delta[FXSizeMax];
	float vega[FXSizeMax];		// Vega
	float rho[FXSizeMax];
	float rhod[FXSizeMax];
	float rhof[FXSizeMax];

	float prob[ScheduleSizeMax];
};