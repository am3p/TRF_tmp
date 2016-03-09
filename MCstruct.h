#pragma once
#include "VariableSize.h"

// Underlying structure
struct Underlying{
	float S;
	
	long RatedType;	// Domestic Rate type: 0 - Fixed, 1 - Term
	long RatedSize;	// Domestic rate size

	long RatefType;	// Foreign Rate type: 0 - Fixed, 1 - Term
	long RatefSize;	// Foreign Rate size

	long VolType;	// Vol type for the underlying: 0 - Fixed, 1 - Term, 2 - Surf
	long VolSizet;	// Vol size along time axis
	long VolSizeK;	// Vol size along price axis
};

// Payoff structure
struct Payoff{
	long T;					// Time to expiry date
	long T_pay;				// Time to payment date

	long BermudanType;		// Bermudan Type: 0 - Final / 1 - Bermudan / 2 - Coupon (in Monthly Redemption Type)

	long PayoffType;		// Payoff Type
							// Vanilla: 0 - Call, 1 - Put (Modified to be compatible in ELS payoffs)
							// Digital: 2 - DigitCall, 3 - DigitPut
							// KO/KI: 4 - KOCall, 5 - KICall, 6 - KOPut, 7 - KIPut
	
	long ObsPriceType;		// Reference price observation option: 0 - Close / 1 - Lowest

	long RefPriceType;		// Reference price setting (btw assets) option: 0 - Minimum / 1 - Average

	float K;				// Strike
	float UpBarrier;		// Up barrier (only in this schedule)
	float DownBarrier;		// Down barrier (only in this schedule)
	float TotalUpBarrier;	// Total up barrier (globally effective)
	float TotalDownBarrier;	// Total down barrier (globally effective)
	float RetUpBarrier;
	float RetDownBarrier;

	float Coupon;			// Coupon amount
	float Dummy;			// Dummy amount, if any
	float Participation;	// Participation rate
};

// YTM structure
struct YTM{
	long YTMType;			// YTM Type	
	long YTMSize;			// YTM Size
};

// Price result
struct Result{
	float price;					// Product price
	float delta[FXSizeMax];		// Delta
	float vega[FXSizeMax];		// Vega
	float rho[FXSizeMax];		// Rho
	float rho_d[FXSizeMax];		// Rho
	float rho_f[FXSizeMax];		// Rho

	long prob;
};