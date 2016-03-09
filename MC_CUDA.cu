//
#include <cuda.h>
#include <curand_kernel.h>

#include "MCstruct.h"
#include "MCstruct_VBA.h"
#include "MCwrapper_fcn.h"
#include "VariableSize.h"

// Global variables for MC
// Underlying: Max 3
__constant__ Underlying Stock[StockSizeMax];
__constant__ double BasePrice[StockSizeMax];

// Schedule: Max 60
__constant__ Payoff Schedule[ScheduleSizeMax];

// YTMinfo: t size 20
__constant__ double YTMt[RateTMax];
__constant__ double YTM[RateTMax];

// Rate: t size 20 per each asset
__constant__ double DRatet[StockSizeMax * RateTMax];
__constant__ double DRate[StockSizeMax * RateTMax];

// Div: t size 20 per each asset
__constant__ double FRatet[StockSizeMax * RateTMax];
__constant__ double FRate[StockSizeMax * RateTMax];

// Vol: t size 20, K size 13 per each asset
__constant__ double Volt[StockSizeMax * VolTMax];
__constant__ double VolK[StockSizeMax * VolKMax];
__constant__ double Vol[StockSizeMax * VolTMax * VolKMax];

// Correlation
__constant__ double correl[StockSizeMax * StockSizeMax];

// Quanto
__constant__ double Quanto[StockSizeMax];

// Global Functions: functions are called in CalcMC
__global__ void InitSeed(curandState *state, const long threadN, const long loopN);
__global__ void MC(curandState *state, 
				   const long StockSize, const long ScheduleSize,
				   const long YTMType, const long YTMSize, double AccRet, const double AccRet_KO,
				   const long SimMode, const long isStrikePriceQuote, const long VolInterpMode, const long threadN, 
				   Result *result);

// Device functions: functions are called in global functions
__device__ double YTMInterp(double t, long YTMType, long YTMSize);	// YTM rate longerp/extrapolation
__device__ double DRfInterp(double t, long stocknum);				// Rf spot rate longerp/extrapolation
__device__ double FRfInterp(double t, long stocknum);				// Rf spot rate longerp/extrapolation
__device__ double VolInterp(double t, double K, long stocknum, long mode);		// Volatility longerp/extrapolationlation

__device__ double SMin(double S_min[][StockSizeMax], long StockSize, long casenum);
__device__ double SMax(double S_max[][StockSizeMax], long StockSize, long casenum);

__device__ double RefPriceCalc(double S, long StockSize, long sched_ind, long casenum);

__device__ bool PayoffCheck(double S[][StockSizeMax], double AccRet, double AccRet_KO, 
							long StockSize, long sched_ind, long casenum);

__device__ double PayoffCalc(double S[][StockSizeMax], double* AccRet, double AccRet_KO, 
							 long StockSize, long sched_ind, long casenum,
							 long isStrikePriceQuote, double participation);

// Main function
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
			long isStrikePriceQuote_, long VolInterpMode_, long SimN_, long SimMode_, long blockN_, long threadN_,
			struct VBAResult* result){

	// GPU parallelization: block/thread for CUDA cores
	long blockN = blockN_;
	long threadN = threadN_;

	// Pseudorandom number state: most simple one provided in CUDA
	curandState *devStates;
	
	// Result vector
	Result *devResults, *hostResults; 
	
	// Allocate space for host vector (4 * 160 is developer's pleases)
	hostResults = (Result *)calloc(blockN * threadN, sizeof(Result)); 
	
	// Allocate space for device vector
	cudaMalloc((void **)&devResults, blockN * threadN * sizeof(Result));  
	cudaMemset(devResults, 0, blockN * threadN * sizeof(Result));	

	// Allocate space for pseudorng states (device) 
	cudaMalloc((void **)&devStates, blockN * threadN * sizeof(curandState));

	// Seed initialization (fixed seed: set to each thread id)

	// Copying product info to global variables
	// Global variable: Stock
	Underlying stock_[StockSizeMax];
	for (long i = 0; i < StockSize_; i++)
	{
		stock_[i].S = StockPrice_[i];
		
		stock_[i].DRateType = DRateType_[i];
		stock_[i].DRateSize = DRateSize_[i];

		stock_[i].FRateType = FRateType_[i];
		stock_[i].FRateSize = FRateSize_[i];

		stock_[i].VolType = VolType_[i];
		stock_[i].VolSizet = VolSizet_[i];
		stock_[i].VolSizeK = VolSizeK_[i];
	}
	Underlying* stock_ptr;
	cudaGetSymbolAddress((void**) &stock_ptr, Stock);
	cudaMemcpy(stock_ptr, stock_, StockSizeMax * sizeof(Underlying), cudaMemcpyHostToDevice);

	// Global variable: YTM
	double* YTMt_ptr;
	cudaGetSymbolAddress((void**) &YTMt_ptr, YTMt);
	cudaMemcpy(YTMt_ptr, YTMt_, RateTMax * sizeof(double), cudaMemcpyHostToDevice);
	double* YTM_ptr;
	cudaGetSymbolAddress((void**) &YTM_ptr, YTM);
	cudaMemcpy(YTM_ptr, YTM_, RateTMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Schedule
	Payoff schedule_[ScheduleSizeMax];
	for (long i = 0; i < ScheduleSize_; i++)
	{
		schedule_[i].T = PayoffT_[i];
		schedule_[i].T_pay = PayoffT_pay[i];
		schedule_[i].BermudanType = BermudanType_[i];
		schedule_[i].PayoffType = PayoffType_[i];
		schedule_[i].RefPriceType = RefPriceType_[i];

		schedule_[i].K = PayoffK_[i];

		schedule_[i].Coupon = Coupon_[i];
		schedule_[i].Dummy = Dummy_[i];

		schedule_[i].Participation = Participation_[i];
	}
	Payoff* sched_ptr;
	cudaGetSymbolAddress((void**) &sched_ptr, Schedule);
	cudaMemcpy(sched_ptr, schedule_, ScheduleSizeMax * sizeof(Payoff), cudaMemcpyHostToDevice);

	double* BasePrice_ptr;
	cudaGetSymbolAddress((void**) &BasePrice_ptr, BasePrice);
	cudaMemcpy(BasePrice_ptr, BasePrice_, StockSizeMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Rate
	double* DRatet_ptr;
	cudaGetSymbolAddress((void**) &DRatet_ptr, DRatet);
	cudaMemcpy(DRatet_ptr, DRatet_, StockSizeMax * RateTMax * sizeof(double), cudaMemcpyHostToDevice);
	double* DRate_ptr;
	cudaGetSymbolAddress((void**) &DRate_ptr, DRate);
	cudaMemcpy(DRate_ptr, DRate_, StockSizeMax * RateTMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Dividend
	double* FRatet_ptr;
	cudaGetSymbolAddress((void**) &FRatet_ptr, FRatet);
	cudaMemcpy(FRatet_ptr, FRatet_, StockSizeMax * RateTMax * sizeof(double), cudaMemcpyHostToDevice);
	double* FRate_ptr;
	cudaGetSymbolAddress((void**) &FRate_ptr, FRate);
	cudaMemcpy(FRate_ptr, FRate_, StockSizeMax * RateTMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Volatility
	double* Volt_ptr;
	cudaGetSymbolAddress((void**) &Volt_ptr, Volt);
	cudaMemcpy(Volt_ptr, Volt_, StockSizeMax * VolTMax * sizeof(double), cudaMemcpyHostToDevice);
	double* VolK_ptr;
	cudaGetSymbolAddress((void**) &VolK_ptr, VolK);
	cudaMemcpy(VolK_ptr, VolK_, StockSizeMax * VolTMax * VolKMax * sizeof(double), cudaMemcpyHostToDevice);
	double* Vol_ptr;
	cudaGetSymbolAddress((void**) &Vol_ptr, Vol);
	cudaMemcpy(Vol_ptr, Vol_, StockSizeMax * VolTMax * VolKMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: correlation
	double* correl_ptr;
	cudaGetSymbolAddress((void **) &correl_ptr, correl);
	cudaMemcpy(correl_ptr, correl_, StockSizeMax * StockSizeMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Quanto
	double* Quanto_ptr;
	cudaGetSymbolAddress((void **) &Quanto_ptr, Quanto);
	cudaMemcpy(Quanto_ptr, Quanto_, StockSizeMax * sizeof(double), cudaMemcpyHostToDevice);

	// Main MC part (the repeat number is just own purpose)
	for (long i = 0; i < SimN_; i++){
		InitSeed<<<blockN, threadN>>>(devStates, threadN, i);
		MC<<<blockN, threadN>>>(devStates, StockSize_, ScheduleSize_, YTMType_, YTMSize_, AccRet, AccRet_KO, SimMode_, isStrikePriceQuote_, VolInterpMode_, threadN, devResults);
		cudaMemcpy(hostResults, devResults, blockN * threadN * sizeof(Result), cudaMemcpyDeviceToHost);

		// Copying MC results
		for (long j = 0; j < blockN * threadN; j++){
			result->price += hostResults[j].price / ((double)(blockN * threadN * SimN_));
			result->prob[hostResults[j].prob] += 1.0 / ((double)(blockN * threadN * SimN_));
			if (SimMode_ > 0){
				for (long k = 0; k < StockSize_; k++){
					result->delta[k] += ((hostResults[j].up_delta[k] + hostResults[j].down_delta[k])/2.0) / ((double)(blockN * threadN * SimN_));
					result->gamma[k] += hostResults[j].gamma[k] / ((double)(blockN * threadN * SimN_));
					result->vega[k] += hostResults[j].vega[k] / ((double)(blockN * threadN * SimN_));
				}
			}
			if (SimMode_ > 1){
				for (long k = 0; k < StockSize_; k++){
					result->rho[k] += hostResults[j].rho[k] / ((double)(blockN * threadN * SimN_));
				}
				result->theta += hostResults[j].theta / ((double)(blockN * threadN * SimN_));
			}
			if (SimMode_ > 2){
				for (long k = 0; k < StockSize_; k++){
					result->vanna[k] += hostResults[j].vanna[k] / ((double)(blockN * threadN * SimN_));
					result->volga[k] += hostResults[j].volga[k] / ((double)(blockN * threadN * SimN_));
				}
			}
		}
	}
	cudaFree(devStates);
	cudaFree(devResults);
	free(hostResults);
}

// Seed initialization
__global__ void InitSeed(curandState *state, const long threadN, const long loopN)
{
	long id = threadIdx.x + blockIdx.x * threadN;
	curand_init(id, 0, loopN * 1000000, &state[id]);
}

// Main Monte Carlo part
__global__ void MC(curandState *state, 
				   const long StockSize, const long ScheduleSize,
				   const long YTMType, const long YTMSize, double AccRet, const double AccRet_KO,
				   const long SimMode, const long isStrikePriceQuote, const long VolInterpMode, const long threadN,
				   Result *result){ 

	// Random number seed ID
	long id = threadIdx.x + blockIdx.x * threadN; 
	// Time interval
	long t = 0; double dt = 1.0/365.0;
	// Necessary number of cash flows
	long CFnum = (long)(pow(2.0, (double)(StockSize+1))-1);
	// Necessary number of adjustments
	long adjnum = (long)(pow(2.0, (double)(StockSize)));
	// Mode: Hardcorded (meaning?)
	long mode = 1;

	// Price variables
	double logS_MC[StockSizeMax], logS_MCmin[StockSizeMax], logS_MCmax[StockSizeMax];
	double logS_MC_Sp[StockSizeMax], logS_MCmin_Sp[StockSizeMax], logS_MCmax_Sp[StockSizeMax];
	double logS_MC_Sm[StockSizeMax], logS_MCmin_Sm[StockSizeMax], logS_MCmax_Sm[StockSizeMax];
	double logS_MC_vp[StockSizeMax], logS_MCmin_vp[StockSizeMax], logS_MCmax_vp[StockSizeMax];
	double logS_MC_vpSp[StockSizeMax], logS_MCmin_vpSp[StockSizeMax], logS_MCmax_vpSp[StockSizeMax];
	double logS_MC_vpSm[StockSizeMax], logS_MCmin_vpSm[StockSizeMax], logS_MCmax_vpSm[StockSizeMax];
	double logS_MC_vm[StockSizeMax], logS_MCmin_vm[StockSizeMax], logS_MCmax_vm[StockSizeMax];
	double logS_MC_vmSp[StockSizeMax], logS_MCmin_vmSp[StockSizeMax], logS_MCmax_vmSp[StockSizeMax];
	double logS_MC_vmSm[StockSizeMax], logS_MCmin_vmSm[StockSizeMax], logS_MCmax_vmSm[StockSizeMax];
	double logS_MC_rp[StockSizeMax], logS_MCmin_rp[StockSizeMax], logS_MCmax_rp[StockSizeMax];
	double logS_MC_tm[StockSizeMax], logS_MCmin_tm[StockSizeMax], logS_MCmax_tm[StockSizeMax];

	// Start point setup
	for (long j = 0; j < StockSize; j++){
		logS_MC[j] = logS_MCmin[j] = logS_MCmax[j] = log(Stock[j].S);
		logS_MC_Sp[j] = logS_MCmin_Sp[j] = logS_MCmax_Sp[j] = log(Stock[j].S * 1.01);
		logS_MC_Sm[j] = logS_MCmin_Sm[j] = logS_MCmax_Sm[j] = log(Stock[j].S * 0.99);

		logS_MC_vp[j] = logS_MCmin_vp[j] = logS_MCmax_vp[j] = log(Stock[j].S);
		logS_MC_vpSp[j] = logS_MCmin_vpSp[j] = logS_MCmax_vpSp[j] = log(Stock[j].S * 1.01);
		logS_MC_vpSm[j] = logS_MCmin_vpSm[j] = logS_MCmax_vpSm[j] = log(Stock[j].S * 0.99);

		logS_MC_vm[j] = logS_MCmin_vm[j] = logS_MCmax_vm[j] = log(Stock[j].S);
		logS_MC_vmSp[j] = logS_MCmin_vmSp[j] = logS_MCmax_vmSp[j] = log(Stock[j].S * 1.01);
		logS_MC_vmSm[j] = logS_MCmin_vmSm[j] = logS_MCmax_vmSm[j] = log(Stock[j].S * 0.99);

		logS_MC_rp[j] = logS_MCmin_rp[j] = logS_MCmax_rp[j] = log(Stock[j].S);
		logS_MC_tm[j] = logS_MCmin_tm[j] = logS_MCmax_tm[j] = log(Stock[j].S);
	}

	// Price information for payoff calculation (current price, min/max along path)
	double S_MC_CF[StockSizeMax], S_MCmin_CF[StockSizeMax], S_MCmax_CF[StockSizeMax];
	double S_MC_CF_Sp[StockSizeMax], S_MCmin_CF_Sp[StockSizeMax], S_MCmax_CF_Sp[StockSizeMax];
	double S_MC_CF_Sm[StockSizeMax], S_MCmin_CF_Sm[StockSizeMax], S_MCmax_CF_Sm[StockSizeMax];
	double S_MC_CF_vp[StockSizeMax], S_MCmin_CF_vp[StockSizeMax], S_MCmax_CF_vp[StockSizeMax];
	double S_MC_CF_vpSp[StockSizeMax], S_MCmin_CF_vpSp[StockSizeMax], S_MCmax_CF_vpSp[StockSizeMax];
	double S_MC_CF_vpSm[StockSizeMax], S_MCmin_CF_vpSm[StockSizeMax], S_MCmax_CF_vpSm[StockSizeMax];
	double S_MC_CF_vm[StockSizeMax], S_MCmin_CF_vm[StockSizeMax], S_MCmax_CF_vm[StockSizeMax];
	double S_MC_CF_vmSp[StockSizeMax], S_MCmin_CF_vmSp[StockSizeMax], S_MCmax_CF_vmSp[StockSizeMax];
	double S_MC_CF_vmSm[StockSizeMax], S_MCmin_CF_vmSm[StockSizeMax], S_MCmax_CF_vmSm[StockSizeMax];
	double S_MC_CF_rp[StockSizeMax], S_MCmin_CF_rp[StockSizeMax], S_MCmax_CF_rp[StockSizeMax];
	double S_MC_CF_tm[StockSizeMax], S_MCmin_CF_tm[StockSizeMax], S_MCmax_CF_tm[StockSizeMax];

	// Payoff calculation for each adjusted stock path
	double S_Payoff[12][StockSizeMax], S_Payoffmin[12][StockSizeMax], S_Payoffmax[12][StockSizeMax];
	
	// Global min/max among all underlyings
	double Smin[12], Smax[12];
	// For range accrual: Up/Down price range
	double logS_RangeUp[StockSizeMax], logS_RangeDown[StockSizeMax];
	for (long i = 0; i < StockSize; i++){
		logS_RangeUp[i] = log(Schedule[0].TotalUpBarrier * BasePrice[i] / 100.0);
		logS_RangeDown[i] = log(Schedule[0].TotalDownBarrier * BasePrice[i] / 100.0);
	}

	// Parameter
	double rfd, rfdp, rfd_fwd, ytm, ytmp, ytmtm, rff, rffp, rff_fwd, vol, volp, volm;

	// Brownian motion variable
	double W_MC_indep[StockSizeMax], W_MC[StockSizeMax];

	// Cash flow status (redeemed or not) and accumulated # of days
	long price_status = 0;						double price_tmp = 0;
	long delta_status[2 * StockSizeMax] = {0};	double delta_tmp[2 * StockSizeMax] = {0};
	long gamma_status[2 * StockSizeMax] = {0};	double gamma_tmp[2 * StockSizeMax] = {0};
	long vega_status[2 * StockSizeMax] = {0};	double vega_tmp[2 * StockSizeMax] = {0};
	long rho_status[StockSizeMax] = {0};		double rho_tmp[StockSizeMax] = {0};
	long theta_status = 0;						double theta_tmp = 0;
	long vanna_status[4 * StockSizeMax] = {0};	double vanna_tmp[4 * StockSizeMax] = {0};
	long volga_status[2 * StockSizeMax] = {0};	double volga_tmp[2 * StockSizeMax] = {0};

	// For range accrual: checking within range or not
	double AccRet_ = AccRet;
	double AccRet_delta[2 * StockSizeMax] = {AccRet,};
	double AccRet_gamma[2 * StockSizeMax] = {AccRet,};
	double AccRet_vega[2 * StockSizeMax] = {AccRet,};
	double AccRet_rho[StockSizeMax] = {AccRet,};
	double AccRet_theta = AccRet;
	double AccRet_vanna[4 * StockSizeMax] = {AccRet,};
	double AccRet_volga[2 * StockSizeMax] = {AccRet,};
	
	// For range accrual: shifted path index and type
	long AccRet_Ind = 0; long AccRet_Type = 0; long AccRet_Type2 = 0;

	// Simulation part
	for(long i = 0; i < ScheduleSize; i++){ 		

		// Innovate until next redemption schedule
		while (t < Schedule[i].T){

			// Generate independent Brownian motion
			for (long j = 0; j < StockSize; j++){
				W_MC_indep[j] = curand_normal(&state[id])*sqrt(dt);
			}
			// Incorporating correlation
			for (long j = StockSize-1; j >= 0; j--){
				W_MC[j] = correl[j*StockSize + j] * W_MC_indep[j];
				for (long k = j-1; k >= 0; k--){
					W_MC[j] += correl[j*StockSize + k] * W_MC_indep[k];
				}
			}
			// Innovation
			for (long j = 0; j < StockSize; j++){

				if (SimMode > 1){
					logS_MC_tm[j] = logS_MC[j];
					logS_MCmin_tm[j] = logS_MCmin[j];
					logS_MCmax_tm[j] = logS_MCmax[j];
				}

				rfd = DRfInterp((double)(t)*dt, j);								// interp/extrap Risk-free rate at t
				rff = FRfInterp((double)(t)*dt, j);								// interp/extrap Risk-free rate at t

				// original path
				vol = VolInterp((double)(t)*dt, exp(logS_MC[j]), j, VolInterpMode);
				logS_MC[j] += (rfd - rff + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];	// Innovation
				logS_MCmin[j] = (logS_MC[j] < logS_MCmin[j]) ? logS_MC[j] : logS_MCmin[j];	// Updating minimum
				logS_MCmax[j] = (logS_MC[j] > logS_MCmax[j]) ? logS_MC[j] : logS_MCmax[j];	// Updating maximum

				if (SimMode > 0){
					// up-shifting price
					vol = VolInterp((double)(t)*dt, expf(logS_MC_Sp[j]), j, VolInterpMode);
					logS_MC_Sp[j] += (rfd - rff + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];				// Innovation
					logS_MCmin_Sp[j] = (logS_MC_Sp[j] < logS_MCmin_Sp[j]) ? logS_MC_Sp[j] : logS_MCmin_Sp[j];	// Updating minimum
					logS_MCmax_Sp[j] = (logS_MC_Sp[j] > logS_MCmax_Sp[j]) ? logS_MC_Sp[j] : logS_MCmax_Sp[j];	// Updating maximum

					// down-shifting price
					vol = VolInterp((double)(t)*dt, expf(logS_MC_Sm[j]), j, VolInterpMode);
					logS_MC_Sm[j] += (rfd - rff + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];				// Innovation
					logS_MCmin_Sm[j] = (logS_MC_Sm[j] < logS_MCmin_Sm[j]) ? logS_MC_Sm[j] : logS_MCmin_Sm[j];	// Updating minimum
					logS_MCmax_Sm[j] = (logS_MC_Sm[j] > logS_MCmax_Sm[j]) ? logS_MC_Sm[j] : logS_MCmax_Sm[j];	// Updating maximum
					

					// up-shifting volatility
					volp = VolInterp((double)(t)*dt, expf(logS_MC_vp[j]), j, VolInterpMode) + 0.01f;
					logS_MC_vp[j] += (rfd - rff + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];												// Innovation
					logS_MCmin_vp[j] = (logS_MC_vp[j] < logS_MCmin_vp[j]) ? logS_MC_vp[j] : logS_MCmin_vp[j];	// Updating minimum
					logS_MCmax_vp[j] = (logS_MC_vp[j] > logS_MCmax_vp[j]) ? logS_MC_vp[j] : logS_MCmax_vp[j];	// Updating maximum
					

					// down-shifting volatility
					volm = VolInterp((double)(t)*dt, expf(logS_MC_vm[j]), j, VolInterpMode) - 0.01f;	
					logS_MC_vm[j] += (rfd - rff + Quanto[j]*volm - volm*volm/2.0f)*dt + volm*W_MC[j];												// Innovation
					logS_MCmin_vm[j] = (logS_MC_vm[j] < logS_MCmin_vm[j]) ? logS_MC_vm[j] : logS_MCmin_vm[j];	// Updating minimum
					logS_MCmax_vm[j] = (logS_MC_vm[j] > logS_MCmax_vm[j]) ? logS_MC_vm[j] : logS_MCmax_vm[j];	// Updating maximum
				}

				if (SimMode > 1){
					// up-shifting risk free rate
					//rfp = rf + 0.001;
					//vol = VolInterp((double)(t)*dt, expf(logS_MC_rp[j]), j, VolInterpMode);
					//logS_MC_rp[j] += (rfdp - rff + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];
					//logS_MCmin_rp[j] = (logS_MC_rp[j] < logS_MCmin_rp[j]) ? logS_MC_rp[j] : logS_MCmin_rp[j];	// Updating minimum
					//logS_MCmax_rp[j] = (logS_MC_rp[j] > logS_MCmax_rp[j]) ? logS_MC_rp[j] : logS_MCmax_rp[j];	// Updating maximum
				}

				if (SimMode > 2){
					volp = VolInterp((double)(t)*dt, expf(logS_MC_vpSp[j]), j, VolInterpMode) + 0.01f;
					// up-shifting volatility, up-shifting price
					logS_MC_vpSp[j] += (rfd - rff + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];						// Innovation
					logS_MCmin_vpSp[j] = (logS_MC_vpSp[j] < logS_MCmin_vpSp[j]) ? logS_MC_vpSp[j] : logS_MCmin_vpSp[j];	// Updating minimum
					logS_MCmax_vpSp[j] = (logS_MC_vpSp[j] > logS_MCmax_vpSp[j]) ? logS_MC_vpSp[j] : logS_MCmax_vpSp[j];	// Updating maximum


					volp = VolInterp((double)(t)*dt, expf(logS_MC_vpSm[j]), j, VolInterpMode) + 0.01f;
					// up-shifting volatility, down-shifting price
					logS_MC_vpSm[j] += (rfd - rff + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];						// Innovation
					logS_MCmin_vpSm[j] = (logS_MC_vpSm[j] < logS_MCmin_vpSm[j]) ? logS_MC_vpSm[j] : logS_MCmin_vpSm[j];	// Updating minimum
					logS_MCmax_vpSm[j] = (logS_MC_vpSm[j] > logS_MCmax_vpSm[j]) ? logS_MC_vpSm[j] : logS_MCmax_vpSm[j];	// Updating maximum
					

					volm = VolInterp((double)(t)*dt, expf(logS_MC_vmSp[j]), j, VolInterpMode) - 0.01f;
					// up-shifting volatility, up-shifting price
					logS_MC_vmSp[j] += (rfd - rff + Quanto[j]*volm - volm*volm*2.0f)*dt + volm*W_MC[j];						// Innovation
					logS_MCmin_vmSp[j] = (logS_MC_vmSp[j] < logS_MCmin_vmSp[j]) ? logS_MC_vmSp[j] : logS_MCmin_vmSp[j];	// Updating minimum
					logS_MCmax_vmSp[j] = (logS_MC_vmSp[j] > logS_MCmax_vmSp[j]) ? logS_MC_vmSp[j] : logS_MCmax_vmSp[j];	// Updating maximum
					

					volm = VolInterp((double)(t)*dt, expf(logS_MC_vmSm[j]), j, VolInterpMode) - 0.01f;
					// up-shifting volatility, down-shifting price
					logS_MC_vmSm[j] += (rfd - rff + Quanto[j]*volm - volm*volm/2.0f)*dt + volm*W_MC[j];						// Innovation
					logS_MCmin_vmSm[j] = (logS_MC_vmSm[j] < logS_MCmin_vmSm[j]) ? logS_MC_vmSm[j] : logS_MCmin_vmSm[j];	// Updating minimum
					logS_MCmax_vmSm[j] = (logS_MC_vmSm[j] > logS_MCmax_vmSm[j]) ? logS_MC_vmSm[j] : logS_MCmax_vmSm[j];	// Updating maximum
				}
			}
			
			__syncthreads();
			t++;
		}

		// Discount rate calculation
		ytm = YTMInterp((double)(Schedule[i].T_pay)*dt, YTMType, YTMSize);
		ytmtm = YTMInterp((double)(Schedule[i].T_pay-1)*dt, YTMType, YTMSize);
		ytmp = ytm + 0.001;

		// Exponantialize stock price; for payoff calculation
		for(long j = 0; j < StockSize; j++){
			S_MC_CF[j] = exp(logS_MC[j]);
			S_MCmin_CF[j] = exp(logS_MCmin[j]);
			S_MCmax_CF[j] = exp(logS_MCmax[j]);
		}

		if (SimMode > 0){
			for (long j = 0; j < StockSize; j++){
				S_MC_CF_Sp[j] = exp(logS_MC_Sp[j]);
				S_MCmin_CF_Sp[j] = exp(logS_MCmin_Sp[j]);
				S_MCmax_CF_Sp[j] = exp(logS_MCmax_Sp[j]);

				S_MC_CF_Sm[j] = exp(logS_MC_Sm[j]);
				S_MCmin_CF_Sm[j] = exp(logS_MCmin_Sm[j]);
				S_MCmax_CF_Sm[j] = exp(logS_MCmax_Sm[j]);

				S_MC_CF_vp[j] = exp(logS_MC_vp[j]);
				S_MCmin_CF_vp[j] = exp(logS_MCmin_vp[j]);
				S_MCmax_CF_vp[j] = exp(logS_MCmax_vp[j]);

				S_MC_CF_vm[j] = exp(logS_MC_vm[j]);
				S_MCmin_CF_vm[j] = exp(logS_MCmin_vm[j]);
				S_MCmax_CF_vm[j] = exp(logS_MCmax_vm[j]);
			}
		}

		if (SimMode > 1){
			for (long j = 0; j < StockSize; j++){
				S_MC_CF_rp[j] = exp(logS_MC_rp[j]);
				S_MCmin_CF_rp[j] = exp(logS_MCmin_rp[j]);
				S_MCmax_CF_rp[j] = exp(logS_MCmax_rp[j]);

				S_MC_CF_tm[j] = exp(logS_MC_tm[j]);
				S_MCmin_CF_tm[j] = exp(logS_MCmin_tm[j]);
				S_MCmax_CF_tm[j] = exp(logS_MCmax_tm[j]);
			}
		}

		if (SimMode > 2){
			for (long j = 0; j < StockSize; j++){
				S_MC_CF_vpSp[j] = exp(logS_MC_vpSp[j]);
				S_MCmin_CF_vpSp[j] = exp(logS_MCmin_vpSp[j]);
				S_MCmax_CF_vpSp[j] = exp(logS_MCmax_vpSp[j]);

				S_MC_CF_vpSm[j] = exp(logS_MC_vpSm[j]);
				S_MCmin_CF_vpSm[j] = exp(logS_MCmin_vpSm[j]);
				S_MCmax_CF_vpSm[j] = exp(logS_MCmax_vpSm[j]);

				S_MC_CF_vmSp[j] = exp(logS_MC_vmSp[j]);
				S_MCmin_CF_vmSp[j] = exp(logS_MCmin_vmSp[j]);
				S_MCmax_CF_vmSp[j] = exp(logS_MCmax_vmSp[j]);

				S_MC_CF_vmSm[j] = exp(logS_MC_vmSm[j]);
				S_MCmin_CF_vmSm[j] = exp(logS_MCmin_vmSm[j]);
				S_MCmax_CF_vmSm[j] = exp(logS_MCmax_vmSm[j]);
			}
		}
			
		// Price
		for (long j = 0; j < StockSize; j++){
			S_Payoff[0][j] = S_MC_CF[j];
			S_Payoffmin[0][j] = S_MCmin_CF[j];
			S_Payoffmax[0][j] = S_MCmax_CF[j];
		}
		Smin[0] = SMin(S_Payoffmin, StockSize, 0);
		Smax[0] = SMax(S_Payoffmax, StockSize, 0);
		if(PayoffCheck(S_Payoff, AccRet_, AccRet_KO, StockSize, i, 0)){					// Checking Redemption
			price_tmp = PayoffCalc(S_Payoff, &AccRet_, AccRet_KO, StockSize, i, 0, isStrikePriceQuote, Schedule[i].Participation) * exp(-ytm*(double)(Schedule[i].T_pay)*dt);
			result[id].prob = i;
		}

			
		//if (SimMode > 0){
		//	// Delta & Gamma
		//	for (long j = 0; j < 2 * StockSize; j++){
		//		for (long k = 0; k < StockSize; k++){
		//			S_Payoff[j][k] = S_MC_CF[k];
		//			S_Payoffmin[j][k] = S_MCmin_CF[k];
		//			S_Payoffmax[j][k] = S_MCmax_CF[k];
		//		}
		//		switch (j){
		//			case 0:
		//				S_Payoff[0][0] = S_MC_CF_Sp[0]; S_Payoffmin[0][0] = S_MCmin_CF_Sp[0]; S_Payoffmax[0][0] = S_MCmax_CF_Sp[0];
		//			case 1:
		//				S_Payoff[1][0] = S_MC_CF_Sm[0]; S_Payoffmin[1][0] = S_MCmin_CF_Sm[0]; S_Payoffmax[1][0] = S_MCmax_CF_Sm[0];
		//			case 2:
		//				S_Payoff[2][1] = S_MC_CF_Sp[1]; S_Payoffmin[2][1] = S_MCmin_CF_Sp[1]; S_Payoffmax[2][1] = S_MCmax_CF_Sp[1];
		//			case 3:
		//				S_Payoff[3][1] = S_MC_CF_Sm[1]; S_Payoffmin[3][1] = S_MCmin_CF_Sm[1]; S_Payoffmax[3][1] = S_MCmax_CF_Sm[1];
		//			case 4:
		//				S_Payoff[4][2] = S_MC_CF_Sp[2]; S_Payoffmin[4][2] = S_MCmin_CF_Sp[2]; S_Payoffmax[4][2] = S_MCmax_CF_Sp[2];
		//			case 5:
		//				S_Payoff[5][2] = S_MC_CF_Sm[2]; S_Payoffmin[5][2] = S_MCmin_CF_Sm[2]; S_Payoffmax[5][2] = S_MCmax_CF_Sm[2];
		//			default:
		//				break;
		//		}
		//	}
		//	for (long j = 0; j < 2 * StockSize; j++){
		//		Smin[j] = SMin(S_Payoffmin, StockSize, j);
		//		Smax[j] = SMax(S_Payoffmax, StockSize, j);
		//	}
		//	for (long j = 0; j < 2*StockSize; j++){
		//		if (delta_status[j] == 0){
		//			if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
		//				delta_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j, isUpTouched, isDownTouched, AccRet_delta[j], isStrikePriceQuote, Schedule[i].Participation) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
		//				(delta_status[j])++;
		//			}
		//		}

		//		if (gamma_status[j] == 0){
		//			if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
		//				gamma_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j, isUpTouched, isDownTouched, AccRet_gamma[j], isStrikePriceQuote, Schedule[i].Participation) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
		//				(gamma_status[j])++;
		//			}
		//		}
		//	}		

		//	// Vega
		//	for (long j = 0; j < 2 * StockSize; j++){
		//		for (long k = 0; k < StockSize; k++){
		//			S_Payoff[j][k] = S_MC_CF[k];
		//			S_Payoffmin[j][k] = S_MCmin_CF[k];
		//			S_Payoffmax[j][k] = S_MCmax_CF[k];
		//		}
		//		switch (j){
		//			case 0:
		//				S_Payoff[0][0] = S_MC_CF_vp[0]; S_Payoffmin[0][0] = S_MCmin_CF_vp[0]; S_Payoffmax[0][0] = S_MCmax_CF_vp[0];
		//			case 1:
		//				S_Payoff[1][0] = S_MC_CF_vm[0]; S_Payoffmin[1][0] = S_MCmin_CF_vm[0]; S_Payoffmax[1][0] = S_MCmax_CF_vm[0];
		//			case 2:
		//				S_Payoff[2][1] = S_MC_CF_vp[1]; S_Payoffmin[2][1] = S_MCmin_CF_vp[1]; S_Payoffmax[2][1] = S_MCmax_CF_vp[1];
		//			case 3:
		//				S_Payoff[3][1] = S_MC_CF_vm[1]; S_Payoffmin[3][1] = S_MCmin_CF_vm[1]; S_Payoffmax[3][1] = S_MCmax_CF_vm[1];
		//			case 4:
		//				S_Payoff[4][2] = S_MC_CF_vp[2]; S_Payoffmin[4][2] = S_MCmin_CF_vp[2]; S_Payoffmax[4][2] = S_MCmax_CF_vp[2];
		//			case 5:
		//				S_Payoff[5][2] = S_MC_CF_vm[2]; S_Payoffmin[5][2] = S_MCmin_CF_vm[2]; S_Payoffmax[5][2] = S_MCmax_CF_vm[2];
		//			default:
		//				break;
		//		}
		//	}
		//	for (long j = 0; j < 2 * StockSize; j++){
		//		Smin[j] = SMin(S_Payoffmin, StockSize, j);
		//		Smax[j] = SMax(S_Payoffmax, StockSize, j);
		//	}
		//	for (long j = 0; j < 2*StockSize; j++){
		//		if (vega_status[j] == 0){
		//			if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
		//				vega_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j, isUpTouched, isDownTouched, AccRet_vega[j], isStrikePriceQuote, Schedule[i].Participation) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
		//				(vega_status[j])++;
		//			}
		//		}
		//	}
		//}

		//if (SimMode > 1){
		//	// Rho
		//	for (long j = 0; j < StockSize; j++){
		//		for (long k = 0; k < StockSize; k++){
		//			S_Payoff[j][k] = S_MC_CF[k];
		//			S_Payoffmin[j][k] = S_MCmin_CF[k];
		//			S_Payoffmax[j][k] = S_MCmax_CF[k];
		//		}
		//		switch (j){
		//			case 0:
		//				S_Payoff[0][0] = S_MC_CF_rp[0]; S_Payoffmin[0][0] = S_MCmin_CF_rp[0]; S_Payoffmax[0][0] = S_MCmax_CF_rp[0];
		//			case 1:
		//				S_Payoff[1][1] = S_MC_CF_rp[1]; S_Payoffmin[1][1] = S_MCmin_CF_rp[1]; S_Payoffmax[1][1] = S_MCmax_CF_rp[1];
		//			case 2:
		//				S_Payoff[2][2] = S_MC_CF_rp[2]; S_Payoffmin[2][2] = S_MCmin_CF_rp[2]; S_Payoffmax[2][2] = S_MCmax_CF_rp[2];
		//			default:
		//				break;
		//		}
		//	}
		//	for (long j = 0; j < StockSize; j++){
		//		Smin[j] = SMin(S_Payoffmin, StockSize, j);
		//		Smax[j] = SMax(S_Payoffmax, StockSize, j);
		//	}
		//	for (long j = 0; j < StockSize; j++){
		//		if (rho_status[j] == 0){
		//			if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
		//				if (j == 0){
		//					rho_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j, isUpTouched, isDownTouched, AccRet_rho[j], isStrikePriceQuote, Schedule[i].Participation) * expf(-ytmp*(double)(Schedule[i].T_pay)*dt);
		//				}
		//				else{
		//					rho_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j, isUpTouched, isDownTouched, AccRet_rho[j], isStrikePriceQuote, Schedule[i].Participation) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
		//				}
		//				(rho_status[j])++;
		//			}
		//		}
		//	}

		//	// Theta
		//	for (long j = 0; j < StockSize; j++){
		//		S_Payoff[0][j] = S_MC_CF_tm[j];
		//		S_Payoffmin[0][j] = S_MCmin_CF_tm[j];
		//		S_Payoffmax[0][j] = S_MCmax_CF_tm[j];
		//	}
		//	for (long j = 0; j < StockSize; j++){
		//		Smin[j] = SMin(S_Payoffmin, StockSize, j);
		//		Smax[j] = SMax(S_Payoffmax, StockSize, j);
		//	}
		//	if (theta_status < 1){
		//		if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, 0)){					// Checking Redemption
		//			theta_tmp = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, 0, isUpTouched, isDownTouched, AccRet_theta, isStrikePriceQuote, Schedule[i].Participation) * expf(-ytmtm*(double)(Schedule[i].T_pay-1)*dt);
		//			theta_status++;
		//		}
		//	}
		//}

		//if (SimMode > 2){
		//	// Vanna
		//	for (long j = 0; j < 4 * StockSize; j++){
		//		for (long k = 0; k < StockSize; k++){
		//			S_Payoff[j][k] = S_MC_CF[k];
		//			S_Payoffmin[j][k] = S_MCmin_CF[k];
		//			S_Payoffmax[j][k] = S_MCmax_CF[k];
		//		}
		//		switch (j){
		//			case 0:
		//				S_Payoff[0][0] = S_MC_CF_vpSp[0]; S_Payoffmin[0][0] = S_MCmin_CF_vpSp[0]; S_Payoffmax[0][0] = S_MCmax_CF_vpSp[0];
		//			case 1:
		//				S_Payoff[1][0] = S_MC_CF_vpSm[0]; S_Payoffmin[1][0] = S_MCmin_CF_vpSm[0]; S_Payoffmax[1][0] = S_MCmax_CF_vpSm[0];
		//			case 2:
		//				S_Payoff[2][0] = S_MC_CF_vmSp[0]; S_Payoffmin[2][0] = S_MCmin_CF_vmSp[0]; S_Payoffmax[2][0] = S_MCmax_CF_vmSp[0];
		//			case 3:
		//				S_Payoff[3][0] = S_MC_CF_vmSm[0]; S_Payoffmin[3][0] = S_MCmin_CF_vmSm[0]; S_Payoffmax[3][0] = S_MCmax_CF_vmSm[0];
		//			case 4:
		//				S_Payoff[4][1] = S_MC_CF_vpSp[1]; S_Payoffmin[4][1] = S_MCmin_CF_vpSp[1]; S_Payoffmax[4][1] = S_MCmax_CF_vpSp[1];
		//			case 5:
		//				S_Payoff[5][1] = S_MC_CF_vpSm[1]; S_Payoffmin[5][1] = S_MCmin_CF_vpSm[1]; S_Payoffmax[5][1] = S_MCmax_CF_vpSm[1];
		//			case 6:
		//				S_Payoff[6][1] = S_MC_CF_vmSp[1]; S_Payoffmin[6][1] = S_MCmin_CF_vmSp[1]; S_Payoffmax[6][1] = S_MCmax_CF_vmSp[1];
		//			case 7:
		//				S_Payoff[7][1] = S_MC_CF_vmSm[1]; S_Payoffmin[7][1] = S_MCmin_CF_vmSm[1]; S_Payoffmax[7][1] = S_MCmax_CF_vmSm[1];
		//			case 8:
		//				S_Payoff[8][2] = S_MC_CF_vpSp[2]; S_Payoffmin[8][2] = S_MCmin_CF_vpSp[2]; S_Payoffmax[8][2] = S_MCmax_CF_vpSp[2];
		//			case 9:
		//				S_Payoff[9][2] = S_MC_CF_vpSm[2]; S_Payoffmin[9][2] = S_MCmin_CF_vpSm[2]; S_Payoffmax[9][2] = S_MCmax_CF_vpSm[2];
		//			case 10:
		//				S_Payoff[10][2] = S_MC_CF_vmSp[2]; S_Payoffmin[10][2] = S_MCmin_CF_vmSp[2]; S_Payoffmax[10][2] = S_MCmax_CF_vmSp[2];
		//			case 11:
		//				S_Payoff[11][2] = S_MC_CF_vmSm[2]; S_Payoffmin[11][2] = S_MCmin_CF_vmSm[2]; S_Payoffmax[11][2] = S_MCmax_CF_vmSm[2];
		//			default:
		//				break;
		//		}
		//	}
		//	for (long j = 0; j < 4 * StockSize; j++){
		//		Smin[j] = SMin(S_Payoffmin, StockSize, j);
		//		Smax[j] = SMax(S_Payoffmax, StockSize, j);
		//	}					
		//	for (long j = 0; j < 4*StockSize; j++){
		//		if (vanna_status[j] == 0){
		//			if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
		//				vanna_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j, isUpTouched, isDownTouched, AccRet_vanna[j], isStrikePriceQuote, Schedule[i].Participation) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
		//				(vanna_status[j])++;
		//			}
		//		}	
		//	}

		//	// Volga
		//	for (long j = 0; j < 2 * StockSize; j++){
		//		for (long k = 0; k < StockSize; k++){
		//			S_Payoff[j][k] = S_MC_CF[k];
		//			S_Payoffmin[j][k] = S_MCmin_CF[k];
		//			S_Payoffmax[j][k] = S_MCmax_CF[k];
		//		}
		//		switch (j){
		//			case 0:
		//				S_Payoff[0][0] = S_MC_CF_vp[0]; S_Payoffmin[0][0] = S_MCmin_CF_vp[0]; S_Payoffmax[0][0] = S_MCmax_CF_vp[0];
		//			case 1:
		//				S_Payoff[1][0] = S_MC_CF_vm[0]; S_Payoffmin[1][0] = S_MCmin_CF_vm[0]; S_Payoffmax[1][0] = S_MCmax_CF_vm[0];
		//			case 2:
		//				S_Payoff[2][1] = S_MC_CF_vp[1]; S_Payoffmin[2][1] = S_MCmin_CF_vp[1]; S_Payoffmax[2][1] = S_MCmax_CF_vp[1];
		//			case 3:
		//				S_Payoff[3][1] = S_MC_CF_vm[1]; S_Payoffmin[3][1] = S_MCmin_CF_vm[1]; S_Payoffmax[3][1] = S_MCmax_CF_vm[1];
		//			case 4:
		//				S_Payoff[4][2] = S_MC_CF_vp[2]; S_Payoffmin[4][2] = S_MCmin_CF_vp[2]; S_Payoffmax[4][2] = S_MCmax_CF_vp[2];
		//			case 5:
		//				S_Payoff[5][2] = S_MC_CF_vm[2]; S_Payoffmin[5][2] = S_MCmin_CF_vm[2]; S_Payoffmax[5][2] = S_MCmax_CF_vm[2];
		//			default:
		//				break;
		//		}
		//	}
		//	for (long j = 0; j < 2 * StockSize; j++){
		//		Smin[j] = SMin(S_Payoffmin, StockSize, j);
		//		Smax[j] = SMax(S_Payoffmax, StockSize, j);
		//	}
		//	for (long j = 0; j < 2*StockSize; j++){
		//		if (volga_status[j] == 0){
		//			if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
		//				volga_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j, isUpTouched, isDownTouched, AccRet_volga[j], isStrikePriceQuote, Schedule[i].Participation) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
		//				(volga_status[j])++;
		//			}
		//		}
		//	}
		//}

	}

	result[id].price = price_tmp;
	if (SimMode > 0){
		for (long i = 0; i < StockSize; i++){
			result[id].up_delta[i] = (delta_tmp[2*i] - price_tmp) / (0.01 * Stock[i].S);
			result[id].down_delta[i] = (price_tmp - delta_tmp[2*i+1]) / (0.01 * Stock[i].S);
		}
		for (long i = 0; i < StockSize; i++)
			result[id].gamma[i] = (gamma_tmp[2*i] - 2.0 * price_tmp + gamma_tmp[2*i+1]) / (0.01 * Stock[i].S * 0.01 * Stock[i].S);
		for (long i = 0; i < StockSize; i++)
			result[id].vega[i] = (vega_tmp[2*i] - vega_tmp[2*i+1]) / 2.0;
	}
	if (SimMode > 1){
		for (long i = 0; i < StockSize; i++)
			result[id].rho[i] = (rho_tmp[i] - price_tmp) / 0.001;
		result[id].theta = price_tmp - theta_tmp;
	}
	if (SimMode > 2){
		for (long i = 0; i < StockSize; i++)
			result[id].vanna[i] = ((vanna_tmp[4*i] - vanna_tmp[4*i+1]) - (vanna_tmp[4*i+2] - vanna_tmp[4*i+3]))/ (2.0f * 2.0f * 0.01f * Stock[i].S);
		for (long i = 0; i < StockSize; i++)
			result[id].volga[i] = (volga_tmp[2*i] - 2.0f * price_tmp + volga_tmp[2*i+1]) / (2.0f * 2.0f);
	}

}


// YTM interp/extrapolation
__device__ double YTMInterp(double t, long YTMType, long YTMSize){
	double r;
	long tind = 0;

	// Fixed case
	if (YTMType == 0)
		r = YTM[0];

	// Term-structure case
	else if (YTMType == 1){
		while (t > YTMt[tind] && tind < YTMSize){
			tind++;
			if (tind == YTMSize)	break;
		}
		
	// nearest extrapolation
		if (tind == 0)				r = YTM[0];
		else if (tind == YTMSize)	r = YTM[YTMSize-1];
		else{
			// linear longerpolation
			r = YTM[tind-1] + 
				(YTM[tind] - YTM[tind-1])/(YTMt[tind] - YTMt[tind-1]) *
				 (t-YTMt[tind-1]);
		}
	}
	return r;
}

// Risk-free rate interp/extrpolation
__device__ double DRfInterp(double t, long stocknum){
	double Rf, Rf_tmp1, Rf_tmp2;
	long tind = 0;

	// Fixed case
	if (Stock[stocknum].DRateType == 0)
		Rf = DRate[stocknum*RateTMax];

	// Term-structure case
	else if (Stock[stocknum].DRateType == 1){
		while (t > DRatet[RateTMax*stocknum + tind] && tind < Stock[stocknum].DRateSize){
			tind++;
			if (tind == Stock[stocknum].DRateSize)	break;
		}
		
		// nearest extrapolation
		if (tind == 0)								Rf = DRate[RateTMax*stocknum];
		else if (tind == Stock[stocknum].DRateSize)	Rf = DRate[RateTMax*stocknum + Stock[stocknum].DRateSize-1];
		else{
			// linear longerpolation
			Rf_tmp1 = DRate[RateTMax*stocknum + tind-1] + 
				      (DRate[RateTMax*stocknum + tind] - DRate[RateTMax*stocknum + tind-1])/(DRatet[RateTMax*stocknum + tind] - DRatet[RateTMax*stocknum + tind-1]) *
				      (t-DRatet[RateTMax*stocknum + tind-1]);
			Rf_tmp2 = DRate[RateTMax*stocknum + tind-1];
			// realized forward rate
			Rf = ((Rf_tmp1 * t) - Rf_tmp2 * DRatet[RateTMax*stocknum + tind-1])/(t-DRatet[RateTMax*stocknum + tind-1]);
		}
	}
	return Rf;
}

// Risk-free rate interp/extrpolation
__device__ double FRfInterp(double t, long stocknum){
	double Rf, Rf_tmp1, Rf_tmp2;
	long tind = 0;

	// Fixed case
	if (Stock[stocknum].FRateType == 0)
		Rf = FRate[stocknum*RateTMax];

	// Term-structure case
	else if (Stock[stocknum].FRateType == 1){
		while (t > FRatet[RateTMax*stocknum + tind] && tind < Stock[stocknum].FRateSize){
			tind++;
			if (tind == Stock[stocknum].FRateSize)	break;
		}
		
		// nearest extrapolation
		if (tind == 0)								Rf = FRate[RateTMax*stocknum];
		else if (tind == Stock[stocknum].FRateSize)	Rf = FRate[RateTMax*stocknum + Stock[stocknum].FRateSize-1];
		else{
			// linear longerpolation
			Rf_tmp1 = FRate[RateTMax*stocknum + tind-1] + 
				      (FRate[RateTMax*stocknum + tind] - FRate[RateTMax*stocknum + tind-1])/(FRatet[RateTMax*stocknum + tind] - FRatet[RateTMax*stocknum + tind-1]) *
				      (t-FRatet[RateTMax*stocknum + tind-1]);
			Rf_tmp2 = FRate[RateTMax*stocknum + tind-1];
			// realized forward rate
			Rf = ((Rf_tmp1 * t) - Rf_tmp2 * FRatet[RateTMax*stocknum + tind-1])/(t-FRatet[RateTMax*stocknum + tind-1]);
		}
	}
	return Rf;
}

__device__ double VolInterp(double t, double K, long stocknum, long mode){
	double v;
	double Vol1, Vol2, Vol11, Vol12, Vol21, Vol22;
	long tind = 0, Kind1 = 0, Kind2 = 0;

	// Fixed case
	if (Stock[stocknum].VolType == 0)
		v = Vol[stocknum*VolTMax*VolKMax];

	// Term structure case (need to be mended!)
	else if (Stock[stocknum].VolType == 1){
		if (t > Volt[VolTMax*stocknum + tind] && tind < Stock[stocknum].VolSizet)
			tind++;
		// nearest extrapolation
		if (tind == 0)								v = Vol[VolTMax*stocknum + 0];
		else if (tind == Stock[stocknum].VolSizet)	v = Vol[VolTMax*stocknum + Stock[stocknum].VolSizet-1];
		else{
			// linear longerpolation
			v = Vol[VolTMax*stocknum + tind-1] + 
				(Vol[VolTMax*stocknum + tind] - Vol[VolTMax*stocknum + tind-1])/(Volt[VolTMax*stocknum + tind] - Volt[VolTMax*stocknum + tind-1]) *
				(t-Volt[VolTMax*stocknum + tind-1]);
		}
	}

	// Surface case
	else if (Stock[stocknum].VolType == 2){

		if (t > Volt[VolTMax*stocknum + tind] && tind < Stock[stocknum].VolSizet){
			while (t > Volt[VolTMax*stocknum + tind] && tind < Stock[stocknum].VolSizet){
				tind++;
				if (tind == Stock[stocknum].VolSizet)	break;
			}
		}

		if (tind == 0){
			if (K > VolK[VolKMax*stocknum + tind*VolKMax + Kind1]){
				while (K > VolK[VolKMax*stocknum + tind*VolKMax + Kind1] && Kind1 < Stock[stocknum].VolSizeK){
					Kind1++;
					if (Kind1 == Stock[stocknum].VolSizeK)	break;
				}
				Kind2 = Kind1;
			}
		}
		else if (tind == Stock[stocknum].VolSizet){
			if (K > VolK[VolKMax*stocknum + (tind-1)*VolKMax + Kind1]){
				while (K > VolK[VolKMax*stocknum + (tind-1)*VolKMax + Kind1] && Kind1 < Stock[stocknum].VolSizeK){
					Kind1++;
					if (Kind1 == Stock[stocknum].VolSizeK)	break;
				}
				Kind2 = Kind1;
			}
		}
		else{
			if (K > VolK[VolKMax*stocknum + (tind-1)*VolKMax + Kind1]){
				while (K > VolK[VolKMax*stocknum + tind*VolKMax + Kind1] && Kind1 < Stock[stocknum].VolSizeK){
					Kind1++;
					if (Kind1 == Stock[stocknum].VolSizeK)	break;
				}				
			}
			if (K > VolK[VolKMax*stocknum + tind*VolKMax + Kind2]){
				while (K > VolK[VolKMax*stocknum + tind*VolKMax + Kind2] && Kind2 < Stock[stocknum].VolSizeK){
					Kind2++;
					if (Kind2 == Stock[stocknum].VolSizeK)	break;
				}
			}
		}



		// Step function along time (1D Linear interpolation along strike)
		if (mode == 0){
			if (tind == Stock[stocknum].VolSizet){
				tind--;
			}

			if (Kind2 == 0)								v = Vol[VolTMax*VolKMax*stocknum + VolKMax*tind + 0];
			else if (Kind2 == Stock[stocknum].VolSizeK)	v = Vol[VolTMax*VolKMax*stocknum + VolKMax*tind + Stock[stocknum].VolSizeK - 1];
			else{
				v = Vol[VolTMax*VolKMax*stocknum + VolKMax*tind + Kind2-1] + 
					(Vol[VolTMax*VolKMax*stocknum + VolKMax*tind + Kind2] - Vol[VolTMax*VolKMax*stocknum + VolKMax*tind + Kind2-1])/(VolK[VolTMax*VolKMax*stocknum + VolKMax*tind + Kind2] - VolK[VolTMax*VolKMax*stocknum + VolKMax*tind + Kind2-1]) *
					(K-VolK[VolTMax*VolKMax*stocknum + VolKMax*tind + Kind2-1]);
			}
		}

		// 2D Linear interpolation
		else if (mode == 1){
			if (tind == 0){
				if (Kind1 == 0)								v = Vol[VolTMax*VolKMax*stocknum + 0];
				else if (Kind1 == Stock[stocknum].VolSizeK)	v = Vol[VolTMax*VolKMax*stocknum + Stock[stocknum].VolSizeK - 1];
				else{
					v = Vol[VolTMax*VolKMax*stocknum + Kind1-1] + 
						(Vol[VolTMax*VolKMax*stocknum + Kind1] - Vol[VolTMax*VolKMax*stocknum + Kind1-1])/(VolK[VolKMax*stocknum + Kind1] - VolK[VolKMax*stocknum + Kind1-1]) *
						(K-VolK[VolKMax*stocknum + Kind1-1]);
				}
			}
			else if (tind == Stock[stocknum].VolSizet){
				if (Kind2 == 0)								v = Vol[VolTMax*VolKMax*stocknum + VolKMax*(Stock[stocknum].VolSizet-1)];
				else if (Kind2 == Stock[stocknum].VolSizeK)	v = Vol[VolTMax*VolKMax*stocknum + VolKMax*(Stock[stocknum].VolSizet-1)+Stock[stocknum].VolSizeK - 1];
				else{
					v = Vol[VolTMax*VolKMax*stocknum + (VolKMax*(Stock[stocknum].VolSizet-1)) + Kind2-1] + 
						(Vol[VolTMax*VolKMax*stocknum + (VolKMax*(Stock[stocknum].VolSizet-1)) + Kind2] - Vol[VolTMax*VolKMax*stocknum + (VolKMax*(Stock[stocknum].VolSizet-1)) + Kind2-1])/(VolK[VolTMax*VolKMax*stocknum + (VolKMax*(Stock[stocknum].VolSizet-1)) + Kind2] - VolK[VolTMax*VolKMax*stocknum + (VolKMax*(Stock[stocknum].VolSizet-1)) + Kind2-1]) *
						(K-VolK[VolTMax*VolKMax*stocknum + (VolKMax*(Stock[stocknum].VolSizet-1)) + Kind2-1]);
				}
			}
			else{
				if (Kind1 == 0){
					Vol1 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind-1)];
				}
				else if (Kind1 == Stock[stocknum].VolSizeK){
					Vol1 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind-1) + Stock[stocknum].VolSizeK-1];
				}
				else{
					Vol11 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind-1) + Kind1-1];
					Vol12 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind-1) + Kind1];
					Vol1 = Vol11 + (Vol12-Vol11)/(VolK[VolKMax*stocknum + Kind1] - VolK[VolKMax*stocknum + Kind1-1]) * (K-VolK[VolKMax*stocknum + Kind1-1]);
				}

				if (Kind2 == 0){
					Vol2 = Vol[VolTMax*VolKMax*stocknum + VolKMax*tind];
				}
				else if (Kind2 == Stock[stocknum].VolSizeK){
					Vol2 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind) + Stock[stocknum].VolSizeK-1];
				}
				else{
					Vol21 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind) + Kind2-1];
					Vol22 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind) + Kind2];
					Vol2 = Vol21 + (Vol22-Vol21)/(VolK[VolKMax*stocknum + Kind2] - VolK[VolKMax*stocknum + Kind2-1]) * (K-VolK[VolKMax*stocknum + Kind2-1]);
				}

				v = Vol1 + (Vol2-Vol1)/(Volt[VolTMax*stocknum + tind] - Volt[VolTMax*stocknum + tind-1]) * (t-Volt[VolTMax*stocknum + tind-1]);
			}
		}

	}
	return v;
}

// Minimum among stock prices
__device__ double SMin(double S_min[][StockSizeMax], long StockSize, long casenum){
	double Min = S_min[casenum][0];
	for (long i = 1; i < StockSize; i++){
		Min = (S_min[casenum][i] < Min) ? S_min[casenum][i] : Min;
	}
	return Min;
}

// Maximum among stock prices
__device__ double SMax(double S_max[][StockSizeMax], long StockSize, long casenum){
	double Max = S_max[casenum][0];
	for (long i = 1; i < StockSize; i++){
		Max = (S_max[casenum][i] > Max) ? S_max[casenum][i] : Max;
	}
	return Max;
}

// Reference price
__device__ double RefPriceCalc(double S[][StockSizeMax], long StockSize, long sched_ind, long casenum){
	double RefPrice = 0;
	switch(Schedule[sched_ind].RefPriceType){
		// Minimum case
		case 0:
			{
				RefPrice = SMin(S, StockSize, casenum);
				break;
			}
		// Average case
		case 1:
			{
				for (long i = 0; i < StockSize; i++){					
					RefPrice += S[casenum][i]/(double)(StockSize);
				}
				break;
			}
		// Maximum case
		case 2:
			{
				RefPrice = SMax(S, StockSize, casenum);
				break;
			}
		default:
			break;
	}
	return RefPrice;
}

// Checking redemption
__device__ bool PayoffCheck(double S[][StockSizeMax], double AccRet, double AccRet_KO, 
							long StockSize, long sched_ind, long casenum){
	bool result = false;
	switch(Schedule[sched_ind].BermudanType){
		// Final case
		case 0:
			{
				switch(Schedule[sched_ind].PayoffType){
					// FORWARD
					case 1:
						{
							if (AccRet < AccRet_KO)		result = true;
							else						result = false;
							break;
						}
					default:
						break;
				}
			}
		// Bermudan case
		case 1:
			{
				switch(Schedule[sched_ind].PayoffType){
					// FORWARD
					case 1:
						{
							if (AccRet < AccRet_KO)		result = true;
							else						result = false;
							break;
						}
					default:
						break;
				}
				break;
			}
		// Coupon case
		case 2:
			{
				result = true;
				break;
			}
		default:
			break;
	}
	return result;
}

// Payoff amount calculation (if redeem)
__device__ double PayoffCalc(double S[][StockSizeMax], double* AccRet, double AccRet_KO, long StockSize, long sched_ind, long casenum,
							 long isStrikePriceQuote, double participation){
	double result = 0;
	switch(Schedule[sched_ind].BermudanType){
		// Final case
		case 0:
			{
				switch(Schedule[sched_ind].PayoffType){
					// FORWARD
					case 1:
						{
							double PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if ((*AccRet) >= AccRet_KO){
								result = 0;
							}
							else{
								result = (PayoffPrice - Schedule[sched_ind].K) / PayoffPrice;
								(*AccRet) += ((PayoffPrice - Schedule[sched_ind].K > 0) ? PayoffPrice - Schedule[sched_ind].K : 0);
							}			
							//result = AccRet_KO;
							break;
						}
					default:
						break;
				}
				break;
			}
		// Bermudan Case
		case 1:
			{
				switch(Schedule[sched_ind].PayoffType){
					// FORWARD
					case 1:
						{
							double PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if ((*AccRet) >= AccRet_KO){
								result = 0;
							}
							else{
								result = (PayoffPrice - Schedule[sched_ind].K) / PayoffPrice;
								(*AccRet) += ((PayoffPrice - Schedule[sched_ind].K > 0) ? PayoffPrice - Schedule[sched_ind].K : 0);
							}	
							//result = AccRet_KO;
							break;
						}
					default:
						break;
				}
				break;
			}
		default:
			break;
	}
	return result;
}