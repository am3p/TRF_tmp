//
#include <cuda.h>
#include <curand_kernel.h>

#include "MCstruct.h"
#include "MCstruct_VBA.h"
#include "MCwrapper_fcn.h"
#include "VariableSize.h"

// Global variables for MC
// Underlying: Max 3
__constant__ Underlying FX[FXSizeMax];
__constant__ float FXBasePrice[FXSizeMax];

// Schedule: Max 60
__constant__ Payoff Schedule[ScheduleSizeMax];

// YTMinfo: t size 20
__constant__ float YTMt[RateTMax];
__constant__ float YTM[RateTMax];

// Rate: t size 20 per each asset
__constant__ float Ratedt[FXSizeMax * RateTMax];
__constant__ float Rated[FXSizeMax * RateTMax];

// Div: t size 20 per each asset
__constant__ float Rateft[FXSizeMax * RateTMax];
__constant__ float Ratef[FXSizeMax * RateTMax];

// Vol: t size 20, K size 13 per each asset
__constant__ float Volt[FXSizeMax * VolTMax];
__constant__ float VolK[FXSizeMax * VolKMax];
__constant__ float Vol[FXSizeMax * VolTMax * VolKMax];

// Correlation
__constant__ float correl[FXSizeMax * FXSizeMax];

// Quanto
__constant__ float Quanto[FXSizeMax];

// Global Functions: functions are called in CalcMC
__global__ void InitSeed(curandState *state, const long threadN);
__global__ void MC(curandState *state, 
				   const long StockSize, const long ScheduleSize, 
				   const long YTMType, const long YTMSize, 
				   const long SimMode, const long isStrikePriceQuote, const long threadN, 
				   Result *result);

// Device functions: functions are called in global functions
__device__ float YTMInterp(float t, long YTMType, long YTMSize);	// YTM rate longerp/extrapolation
__device__ float RatedInterp(float t, long fxnum);					// Rf spot rate longerp/extrapolation
__device__ float RatefInterp(float t, long fxnum);					// Rf spot rate longerp/extrapolation
__device__ float DivInterp(float t, long fxnum);				// Dividend longerp/extrapolation
__device__ float VolInterp(float t, float K, long fxnum);		// Volatility longerp/extrapolationlation

__device__ float FXMin(float FX_min[][FXSizeMax], long FXSize, long casenum);
__device__ float FXMax(float FX_max[][FXSizeMax], long FXSize, long casenum);

__device__ float RefPriceCalc(float S, long StockSize, long sched_ind, long casenum);
__device__ bool PayoffCheck(float S[][FXSizeMax], float* S_min, float* S_max, long StockSize, long sched_ind, long casenum);
__device__ float PayoffCalc(float S[][FXSizeMax], float* S_min, float* S_max, long StockSize, long sched_ind, long casenum);

// Main function
void CalcMC(float* FXPrice_, long FXSize_,
			long ScheduleSize_,	
			long* PayoffT_, long* PayoffT_pay, long* PayoffType_, long* RefPriceType_,
			float* PayoffK_, float Accret_,
 			long* RatedType_, long* RatedSize_, float* Ratedt_, float* Rated_,
			long* RatefType_, long* RatefSize_, float* Rateft_, float* Ratef_,
 			long* VolType_, long* VolSizet_, long* VolSizeK_, float* Volt_, float* VolK_, float* Vol_,
			long YTMType_, long YTMSize_, float* YTMt_, float* YTM_,
			long isStrikePriceQuote_, long SimN_, long SimMode_, long blockN_, long threadN_,
			struct VBAResult* result){
	
	// GPU parallelization: block/thread for CUDA cores
	long blockN = blockN_;
	long threadN = threadN_;

	// Pseudorandom number state: most simple one provided in CUDA
	curandState *devStates;
	
	// Result vector
	Result *devResults, *hostResults; 
	
	// Allocate space for host vector
	hostResults = (Result *)calloc(blockN * threadN, sizeof(Result)); 
	
	// Allocate space for device vector
	cudaMalloc((void **)&devResults, blockN * threadN * sizeof(Result));  
	cudaMemset(devResults, 0, blockN * threadN * sizeof(Result));	

	// Allocate space for pseudorng states (device) 
	cudaMalloc((void **)&devStates, blockN * threadN * sizeof(curandState));

	// Seed initialization (fixed seed: set to each thread id)
	InitSeed<<<blockN, threadN>>>(devStates, threadN);

	// Copying product info to global variables
	// Global variable: Stock
	Underlying FX_[FXSizeMax];
	for (long i = 0; i < FXSize_; i++)
	{
		FX_[i].S = FXPrice_[i];
		
		FX_[i].RatedType = RatedType_[i];
		FX_[i].RatedSize = RatedSize_[i];

		FX_[i].RatefType = RatefType_[i];
		FX_[i].RatefSize = RatefSize_[i];

		FX_[i].VolType = VolType_[i];
		FX_[i].VolSizet = VolSizet_[i];
		FX_[i].VolSizeK = VolSizeK_[i];
	}
	Underlying* FX_ptr;
	cudaGetSymbolAddress((void**) &FX_ptr, FX);
	cudaMemcpy(FX_ptr, FX_, FXSizeMax * sizeof(Underlying), cudaMemcpyHostToDevice);

	// Global variable: YTM
	float* YTMt_ptr;
	cudaGetSymbolAddress((void**) &YTMt_ptr, YTMt);
	cudaMemcpy(YTMt_ptr, YTMt_, RateTMax * sizeof(float), cudaMemcpyHostToDevice);
	float* YTM_ptr;
	cudaGetSymbolAddress((void**) &YTM_ptr, YTM);
	cudaMemcpy(YTM_ptr, YTM_, RateTMax * sizeof(float), cudaMemcpyHostToDevice);

	// Global variable: Schedule
	Payoff schedule_[ScheduleSizeMax];
	for (long i = 0; i < ScheduleSize_; i++)
	{
		schedule_[i].T = PayoffT_[i];
		schedule_[i].T_pay = PayoffT_pay[i];
		schedule_[i].PayoffType = PayoffType_[i];
		schedule_[i].RefPriceType = RefPriceType_[i];

		schedule_[i].K = PayoffK_[i];
	}
	Payoff* sched_ptr;
	cudaGetSymbolAddress((void**) &sched_ptr, Schedule);
	cudaMemcpy(sched_ptr, schedule_, ScheduleSizeMax * sizeof(Payoff), cudaMemcpyHostToDevice);

	// Global variable: Rate
	float* Ratedt_ptr;
	cudaGetSymbolAddress((void**) &Ratedt_ptr, Ratedt);
	cudaMemcpy(Ratedt_ptr, Ratedt_, FXSizeMax * RateTMax * sizeof(float), cudaMemcpyHostToDevice);
	float* Rated_ptr;
	cudaGetSymbolAddress((void**) &Rated_ptr, Rated);
	cudaMemcpy(Rated_ptr, Rated_, FXSizeMax * RateTMax * sizeof(float), cudaMemcpyHostToDevice);

	// Global variable: Dividend
	float* Rateft_ptr;
	cudaGetSymbolAddress((void**) &Rateft_ptr, Rateft);
	cudaMemcpy(Rateft_ptr, Rateft_, FXSizeMax * RateTMax * sizeof(float), cudaMemcpyHostToDevice);
	float* Ratef_ptr;
	cudaGetSymbolAddress((void**) &Ratef_ptr, Ratef);
	cudaMemcpy(Ratef_ptr, Ratef_, FXSizeMax * RateTMax * sizeof(float), cudaMemcpyHostToDevice);

	// Global variable: Volatility
	float* Volt_ptr;
	cudaGetSymbolAddress((void**) &Volt_ptr, Volt);
	cudaMemcpy(Volt_ptr, Volt_, FXSizeMax * VolTMax * sizeof(float), cudaMemcpyHostToDevice);
	float* VolK_ptr;
	cudaGetSymbolAddress((void**) &VolK_ptr, VolK);
	cudaMemcpy(VolK_ptr, VolK_, FXSizeMax * VolKMax * sizeof(float), cudaMemcpyHostToDevice);
	float* Vol_ptr;
	cudaGetSymbolAddress((void**) &Vol_ptr, Vol);
	cudaMemcpy(Vol_ptr, Vol_, FXSizeMax * VolTMax * VolKMax * sizeof(float), cudaMemcpyHostToDevice);

	// Main MC part (the repeat number is just own purpose)
	for (long i = 0; i < SimN_; i++){
		MC<<<blockN, threadN>>>(devStates, FXSize_, ScheduleSize_, YTMType_, YTMSize_, SimMode_, isStrikePriceQuote_, threadN, devResults);
		cudaMemcpy(hostResults, devResults, blockN * threadN * sizeof(Result), cudaMemcpyDeviceToHost);

		// Copying MC results
		for (long j = 0; j < blockN * threadN; j++){
			result->price += hostResults[j].price / ((float)(blockN * threadN * SimN_));
			result->prob[hostResults[j].prob] += 1.0f / ((float)(blockN * threadN * SimN_));
			if (SimMode_ > 0){
				for (long k = 0; k < 1; k++){
					result->delta[k] += hostResults[j].delta[k] / ((float)(blockN * threadN * SimN_));
					//result->gamma[k] += hostResults[j].gamma[k] / ((float)(blockN * threadN * SimN_));
					result->vega[k] += hostResults[j].vega[k] / ((float)(blockN * threadN * SimN_));
				}
			}
			if (SimMode_ > 1){
				for (long k = 0; k < 1; k++){
					//result->rho[k] += hostResults[j].rho[k] / ((float)(blockN * threadN * SimN_));
				}
				//result->theta += hostResults[j].theta / ((float)(blockN * threadN * SimN_));
			}
			if (SimMode_ > 2){
				for (long k = 0; k < 1; k++){
					//result->vanna[k] += hostResults[j].vanna[k] / ((float)(blockN * threadN * SimN_));
					//result->volga[k] += hostResults[j].volga[k] / ((float)(blockN * threadN * SimN_));
				}
			}
		}
	}
	cudaFree(devStates);
	cudaFree(devResults);
	free(hostResults);
}

// Seed initialization
__global__ void InitSeed(curandState *state, const long threadN)
{
	long id = threadIdx.x + blockIdx.x * threadN;
	curand_init(id, 0, 0, &state[id]);
}

// Main Monte Carlo part
__global__ void MC(curandState *state, 
				   const long FXSize, const long ScheduleSize, 
				   const long YTMType, const long YTMSize, 
				   const long SimMode, const long isStrikePriceQuote, const long threadN,
				   Result *result){ 
	long id = threadIdx.x + blockIdx.x * threadN; 
	long t = 0; float dt = 1.0f/365.0f;
	long CFnum = (long)(powf(2.0f, (float)(FXSize+1))-1);
	long adjnum = (long)(powf(2.0f, (float)(FXSize)));

	float accret = 0.0f;

	// Price variables
	float logS_MC[FXSizeMax], logS_MCmin[FXSizeMax], logS_MCmax[FXSizeMax];
	float logS_MC_Sp[FXSizeMax], logS_MCmin_Sp[FXSizeMax], logS_MCmax_Sp[FXSizeMax];
	float logS_MC_Sm[FXSizeMax], logS_MCmin_Sm[FXSizeMax], logS_MCmax_Sm[FXSizeMax];
	float logS_MC_vp[FXSizeMax], logS_MCmin_vp[FXSizeMax], logS_MCmax_vp[FXSizeMax];
	float logS_MC_vm[FXSizeMax], logS_MCmin_vm[FXSizeMax], logS_MCmax_vm[FXSizeMax];
	float logS_MC_rp[FXSizeMax], logS_MCmin_rp[FXSizeMax], logS_MCmax_rp[FXSizeMax];
	float logS_MC_rm[FXSizeMax], logS_MCmin_rm[FXSizeMax], logS_MCmax_rm[FXSizeMax];
	float logS_MC_tm[FXSizeMax], logS_MCmin_tm[FXSizeMax], logS_MCmax_tm[FXSizeMax];

	for (long j = 0; j < FXSize; j++){
		logS_MC[j] = logS_MCmin[j] = logS_MCmax[j] = logf(FX[j].S);
		logS_MC_Sp[j] = logS_MCmin_Sp[j] = logS_MCmax_Sp[j] = logf(FX[j].S * 1.01f);
		logS_MC_Sm[j] = logS_MCmin_Sm[j] = logS_MCmax_Sm[j] = logf(FX[j].S * 0.99f);

		logS_MC_vp[j] = logS_MCmin_vp[j] = logS_MCmax_vp[j] = logf(FX[j].S);
		logS_MC_vm[j] = logS_MCmin_vm[j] = logS_MCmax_vm[j] = logf(FX[j].S);

		logS_MC_rp[j] = logS_MCmin_rp[j] = logS_MCmax_rp[j] = logf(FX[j].S);
		logS_MC_rm[j] = logS_MCmin_rm[j] = logS_MCmax_rm[j] = logf(FX[j].S);

		logS_MC_tm[j] = logS_MCmin_tm[j] = logS_MCmax_tm[j] = logf(FX[j].S);
	}

	// Price information for payoff calculation (current price, min/max along path)
	float S_MC_CF[FXSizeMax], S_MCmin_CF[FXSizeMax], S_MCmax_CF[FXSizeMax];
	float S_MC_CF_Sp[FXSizeMax], S_MCmin_CF_Sp[FXSizeMax], S_MCmax_CF_Sp[FXSizeMax];
	float S_MC_CF_Sm[FXSizeMax], S_MCmin_CF_Sm[FXSizeMax], S_MCmax_CF_Sm[FXSizeMax];
	float S_MC_CF_vp[FXSizeMax], S_MCmin_CF_vp[FXSizeMax], S_MCmax_CF_vp[FXSizeMax];
	float S_MC_CF_vm[FXSizeMax], S_MCmin_CF_vm[FXSizeMax], S_MCmax_CF_vm[FXSizeMax];
	float S_MC_CF_rp[FXSizeMax], S_MCmin_CF_rp[FXSizeMax], S_MCmax_CF_rp[FXSizeMax];
	float S_MC_CF_rm[FXSizeMax], S_MCmin_CF_rm[FXSizeMax], S_MCmax_CF_rm[FXSizeMax];
	float S_MC_CF_tm[FXSizeMax], S_MCmin_CF_tm[FXSizeMax], S_MCmax_CF_tm[FXSizeMax];

	float S_Payoff[10][FXSizeMax], S_Payoffmin[10][FXSizeMax], S_Payoffmax[10][FXSizeMax];
	
	// Global min/max among all underlyings
	float Smin[10], Smax[10];
	// Parameter
	float rd, rdp, rdm, rf, rfp, rfm, ytm, ytmp, ytmm, ytmtm, vol, volp, volm;

	// Brownian motion variable
	float W_MC_indep[FXSizeMax], W_MC[FXSizeMax];

	// Cash flow status (redeemed or not)
	long price_status = 0;						float price_tmp = 0;
	long delta_status[2 * FXSizeMax] = {0};	float delta_tmp[2 * FXSizeMax] = {0};
	long vega_status[2 * FXSizeMax] = {0};	float vega_tmp[2 * FXSizeMax] = {0};
	long rho_status[FXSizeMax] = {0};		float rho_tmp[FXSizeMax] = {0};
	long theta_status = 0;						float theta_tmp = 0;

	// Simulation part
	for(long i = 0; i < ScheduleSize; i++){ 
		// Innovate until next redemption schedule
		while (t < Schedule[i].T){
			// Generate independent Brownian motion
			for (long j = 0; j < FXSize; j++){
				W_MC_indep[j] = curand_normal(&state[id])*sqrt(dt);
			}
			// Incorporating correlation
			for (long j = FXSize-1; j >= 0; j--){
				W_MC[j] = correl[j*FXSize + j] * W_MC_indep[j];
				for (long k = j-1; k >= 0; k--){
					W_MC[j] += correl[j*FXSize + k] * W_MC_indep[k];
				}
			}
			// Innovation
			for (long j = 0; j < FXSize; j++){

				if (SimMode > 1){
					logS_MC_tm[j] = logS_MC[j];
					logS_MCmin_tm[j] = logS_MCmin[j];
					logS_MCmax_tm[j] = logS_MCmax[j];
				}

				rd = RatedInterp((float)(t)*dt, j);
				rf = RatefInterp((float)(t)*dt, j);							// longerp/extrap Risk-free rate at t

				// original path
				vol = VolInterp((float)(t)*dt, expf(logS_MC[j]), j);
				logS_MC[j] += (rd - rf - vol*vol/2.0f)*dt + vol*W_MC[j];	// Innovation
				logS_MCmin[j] = (logS_MC[j] < logS_MCmin[j]) ? logS_MC[j] : logS_MCmin[j];	// Updating minimum
				logS_MCmax[j] = (logS_MC[j] > logS_MCmax[j]) ? logS_MC[j] : logS_MCmax[j];	// Updating maximum

				if (SimMode > 0){
					// up-shifting price
					vol = VolInterp((float)(t)*dt, expf(logS_MC_Sp[j]), j);
					logS_MC_Sp[j] += (rd - rf - vol*vol/2.0f)*dt + vol*W_MC[j];				// Innovation
					logS_MCmin_Sp[j] = (logS_MC_Sp[j] < logS_MCmin_Sp[j]) ? logS_MC_Sp[j] : logS_MCmin_Sp[j];	// Updating minimum
					logS_MCmax_Sp[j] = (logS_MC_Sp[j] > logS_MCmax_Sp[j]) ? logS_MC_Sp[j] : logS_MCmax_Sp[j];	// Updating maximum

					// down-shifting price
					vol = VolInterp((float)(t)*dt, expf(logS_MC_Sm[j]), j);
					logS_MC_Sm[j] += (rd - rf - vol*vol/2.0f)*dt + vol*W_MC[j];				// Innovation
					logS_MCmin_Sm[j] = (logS_MC_Sm[j] < logS_MCmin_Sm[j]) ? logS_MC_Sm[j] : logS_MCmin_Sm[j];	// Updating minimum
					logS_MCmax_Sm[j] = (logS_MC_Sm[j] > logS_MCmax_Sm[j]) ? logS_MC_Sm[j] : logS_MCmax_Sm[j];	// Updating maximum

					// up-shifting volatility
					volp = VolInterp((float)(t)*dt, expf(logS_MC_vp[j]), j) + 0.01f;
					logS_MC_vp[j] += (rd - rf - volp*volp/2.0f)*dt + volp*W_MC[j];												// Innovation
					logS_MCmin_vp[j] = (logS_MC_vp[j] < logS_MCmin_vp[j]) ? logS_MC_vp[j] : logS_MCmin_vp[j];	// Updating minimum
					logS_MCmax_vp[j] = (logS_MC_vp[j] > logS_MCmax_vp[j]) ? logS_MC_vp[j] : logS_MCmax_vp[j];	// Updating maximum

					// down-shifting volatility
					volm = VolInterp((float)(t)*dt, expf(logS_MC_vm[j]), j) - 0.01f;	
					logS_MC_vm[j] += (rd - rf - volm*volm/2.0f)*dt + volm*W_MC[j];												// Innovation
					logS_MCmin_vm[j] = (logS_MC_vm[j] < logS_MCmin_vm[j]) ? logS_MC_vm[j] : logS_MCmin_vm[j];	// Updating minimum
					logS_MCmax_vm[j] = (logS_MC_vm[j] > logS_MCmax_vm[j]) ? logS_MC_vm[j] : logS_MCmax_vm[j];	// Updating maximum
				}

				if (SimMode > 1){
					// up-shifting foreign risk free rate: 10bp
					vol = VolInterp((float)(t)*dt, expf(logS_MC_rp[j]), j);
					logS_MC_rp[j] += (rd - rf + 0.001 - vol*vol/2.0f)*dt + vol*W_MC[j];
					logS_MCmin_rp[j] = (logS_MC_rp[j] < logS_MCmin_rp[j]) ? logS_MC_rp[j] : logS_MCmin_rp[j];	// Updating minimum
					logS_MCmax_rp[j] = (logS_MC_rp[j] > logS_MCmax_rp[j]) ? logS_MC_rp[j] : logS_MCmax_rp[j];	// Updating maximum

					// down-shifting foreign risk free rate: 10bp
					vol = VolInterp((float)(t)*dt, expf(logS_MC_rm[j]), j);
					logS_MC_rm[j] += (rd - rf - 0.001 - vol*vol/2.0f)*dt + vol*W_MC[j];
					logS_MCmin_rm[j] = (logS_MC_rm[j] < logS_MCmin_rm[j]) ? logS_MC_rm[j] : logS_MCmin_rm[j];	// Updating minimum
					logS_MCmax_rm[j] = (logS_MC_rm[j] > logS_MCmax_rm[j]) ? logS_MC_rm[j] : logS_MCmax_rm[j];	// Updating maximum
				}

				if (SimMode > 2){
					//volp = VolInterp((float)(t)*dt, expf(logS_MC_vpSp[j]), j) + 0.01f;
					//// up-shifting volatility, up-shifting price
					//logS_MC_vpSp[j] += (rf - div + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];						// Innovation
					//logS_MCmin_vpSp[j] = (logS_MC_vpSp[j] < logS_MCmin_vpSp[j]) ? logS_MC_vpSp[j] : logS_MCmin_vpSp[j];	// Updating minimum
					//logS_MCmax_vpSp[j] = (logS_MC_vpSp[j] > logS_MCmax_vpSp[j]) ? logS_MC_vpSp[j] : logS_MCmax_vpSp[j];	// Updating maximum

					//volp = VolInterp((float)(t)*dt, expf(logS_MC_vpSm[j]), j) + 0.01f;
					//// up-shifting volatility, down-shifting price
					//logS_MC_vpSm[j] += (rf - div + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];						// Innovation
					//logS_MCmin_vpSm[j] = (logS_MC_vpSm[j] < logS_MCmin_vpSm[j]) ? logS_MC_vpSm[j] : logS_MCmin_vpSm[j];	// Updating minimum
					//logS_MCmax_vpSm[j] = (logS_MC_vpSm[j] > logS_MCmax_vpSm[j]) ? logS_MC_vpSm[j] : logS_MCmax_vpSm[j];	// Updating maximum

					//volm = VolInterp((float)(t)*dt, expf(logS_MC_vmSp[j]), j) - 0.01f;
					//// up-shifting volatility, up-shifting price
					//logS_MC_vmSp[j] += (rf - div + Quanto[j]*volm - volm*volm*2.0f)*dt + volm*W_MC[j];						// Innovation
					//logS_MCmin_vmSp[j] = (logS_MC_vmSp[j] < logS_MCmin_vmSp[j]) ? logS_MC_vmSp[j] : logS_MCmin_vmSp[j];	// Updating minimum
					//logS_MCmax_vmSp[j] = (logS_MC_vmSp[j] > logS_MCmax_vmSp[j]) ? logS_MC_vmSp[j] : logS_MCmax_vmSp[j];	// Updating maximum

					//volm = VolInterp((float)(t)*dt, expf(logS_MC_vmSm[j]), j) - 0.01f;
					//// up-shifting volatility, down-shifting price
					//logS_MC_vmSm[j] += (rf - div + Quanto[j]*volm - volm*volm/2.0f)*dt + volm*W_MC[j];						// Innovation
					//logS_MCmin_vmSm[j] = (logS_MC_vmSm[j] < logS_MCmin_vmSm[j]) ? logS_MC_vmSm[j] : logS_MCmin_vmSm[j];	// Updating minimum
					//logS_MCmax_vmSm[j] = (logS_MC_vmSm[j] > logS_MCmax_vmSm[j]) ? logS_MC_vmSm[j] : logS_MCmax_vmSm[j];	// Updating maximum
				}
			}
			__syncthreads();
			t++;
		}
		ytm = YTMInterp((float)(Schedule[i].T_pay)*dt, YTMType, YTMSize);
		ytmtm = YTMInterp((float)(Schedule[i].T_pay-1)*dt, YTMType, YTMSize);
		ytmp = ytm + 0.001;
		ytmm = ytm - 0.001;

		for(long j = 0; j < FXSize; j++){
			if (isStrikePriceQuote == 1){
				S_MC_CF[j] = expf(logS_MC[j]);
				S_MCmin_CF[j] = expf(logS_MCmin[j]);
				S_MCmax_CF[j] = expf(logS_MCmax[j]);
			}
			else if (isStrikePriceQuote == 0){
				S_MC_CF[j] = expf(logS_MC[j])/FXBasePrice[j] * 100.0f;
				S_MCmin_CF[j] = expf(logS_MCmin[j])/FXBasePrice[j] * 100.0f;
				S_MCmax_CF[j] = expf(logS_MCmax[j])/FXBasePrice[j] * 100.0f;
			}
		}

		if (SimMode > 0){
			if (isStrikePriceQuote == 1){
				for (long j = 0; j < FXSize; j++){
					S_MC_CF_Sp[j] = expf(logS_MC_Sp[j]);
					S_MCmin_CF_Sp[j] = expf(logS_MCmin_Sp[j]);
					S_MCmax_CF_Sp[j] = expf(logS_MCmax_Sp[j]);

					S_MC_CF_Sm[j] = expf(logS_MC_Sm[j]);
					S_MCmin_CF_Sm[j] = expf(logS_MCmin_Sm[j]);
					S_MCmax_CF_Sm[j] = expf(logS_MCmax_Sm[j]);

					S_MC_CF_vp[j] = expf(logS_MC_vp[j]);
					S_MCmin_CF_vp[j] = expf(logS_MCmin_vp[j]);
					S_MCmax_CF_vp[j] = expf(logS_MCmax_vp[j]);

					S_MC_CF_vm[j] = expf(logS_MC_vm[j]);
					S_MCmin_CF_vm[j] = expf(logS_MCmin_vm[j]);
					S_MCmax_CF_vm[j] = expf(logS_MCmax_vm[j]);
				}
			}
			else if (isStrikePriceQuote == 0){
				for (long j = 0; j < FXSize; j++){
					S_MC_CF_Sp[j] = expf(logS_MC_Sp[j])/FXBasePrice[j] * 100.0f;
					S_MCmin_CF_Sp[j] = expf(logS_MCmin_Sp[j])/FXBasePrice[j] * 100.0f;
					S_MCmax_CF_Sp[j] = expf(logS_MCmax_Sp[j])/FXBasePrice[j] * 100.0f;

					S_MC_CF_Sm[j] = expf(logS_MC_Sm[j])/FXBasePrice[j] * 100.0f;
					S_MCmin_CF_Sm[j] = expf(logS_MCmin_Sm[j])/FXBasePrice[j] * 100.0f;
					S_MCmax_CF_Sm[j] = expf(logS_MCmax_Sm[j])/FXBasePrice[j] * 100.0f;

					S_MC_CF_vp[j] = expf(logS_MC_vp[j])/FXBasePrice[j] * 100.0f;
					S_MCmin_CF_vp[j] = expf(logS_MCmin_vp[j])/FXBasePrice[j] * 100.0f;
					S_MCmax_CF_vp[j] = expf(logS_MCmax_vp[j])/FXBasePrice[j] * 100.0f;

					S_MC_CF_vm[j] = expf(logS_MC_vm[j])/FXBasePrice[j] * 100.0f;
					S_MCmin_CF_vm[j] = expf(logS_MCmin_vm[j])/FXBasePrice[j] * 100.0f;
					S_MCmax_CF_vm[j] = expf(logS_MCmax_vm[j])/FXBasePrice[j] * 100.0f;
				}
			}
		}

		if (SimMode > 1){
			if (isStrikePriceQuote == 1){
				for (long j = 0; j < FXSize; j++){
					S_MC_CF_rp[j] = expf(logS_MC_rp[j]);
					S_MCmin_CF_rp[j] = expf(logS_MCmin_rp[j]);
					S_MCmax_CF_rp[j] = expf(logS_MCmax_rp[j]);

					S_MC_CF_rm[j] = expf(logS_MC_rm[j]);
					S_MCmin_CF_rm[j] = expf(logS_MCmin_rm[j]);
					S_MCmax_CF_rm[j] = expf(logS_MCmax_rm[j]);

					S_MC_CF_tm[j] = expf(logS_MC_tm[j]);
					S_MCmin_CF_tm[j] = expf(logS_MCmin_tm[j]);
					S_MCmax_CF_tm[j] = expf(logS_MCmax_tm[j]);
				}
			}
			else if (isStrikePriceQuote == 0){
				for (long j = 0; j < FXSize; j++){
					S_MC_CF_rp[j] = expf(logS_MC_rp[j])/FXBasePrice[j] * 100.0f;
					S_MCmin_CF_rp[j] = expf(logS_MCmin_rp[j])/FXBasePrice[j] * 100.0f;
					S_MCmax_CF_rp[j] = expf(logS_MCmax_rp[j])/FXBasePrice[j] * 100.0f;

					S_MC_CF_rm[j] = expf(logS_MC_rm[j])/FXBasePrice[j] * 100.0f;
					S_MCmin_CF_rm[j] = expf(logS_MCmin_rm[j])/FXBasePrice[j] * 100.0f;
					S_MCmax_CF_rm[j] = expf(logS_MCmax_rm[j])/FXBasePrice[j] * 100.0f;

					S_MC_CF_tm[j] = expf(logS_MC_tm[j])/FXBasePrice[j] * 100.0f;
					S_MCmin_CF_tm[j] = expf(logS_MCmin_tm[j])/FXBasePrice[j] * 100.0f;
					S_MCmax_CF_tm[j] = expf(logS_MCmax_tm[j])/FXBasePrice[j] * 100.0f;
				}
			}
		}

		if (SimMode > 2){
			//if (isStrikePriceQuote == 1){
			//	for (long j = 0; j < StockSize; j++){
			//		S_MC_CF_vpSp[j] = expf(logS_MC_vpSp[j]);
			//		S_MCmin_CF_vpSp[j] = expf(logS_MCmin_vpSp[j]);
			//		S_MCmax_CF_vpSp[j] = expf(logS_MCmax_vpSp[j]);

			//		S_MC_CF_vpSm[j] = expf(logS_MC_vpSm[j]);
			//		S_MCmin_CF_vpSm[j] = expf(logS_MCmin_vpSm[j]);
			//		S_MCmax_CF_vpSm[j] = expf(logS_MCmax_vpSm[j]);

			//		S_MC_CF_vmSp[j] = expf(logS_MC_vmSp[j]);
			//		S_MCmin_CF_vmSp[j] = expf(logS_MCmin_vmSp[j]);
			//		S_MCmax_CF_vmSp[j] = expf(logS_MCmax_vmSp[j]);

			//		S_MC_CF_vmSm[j] = expf(logS_MC_vmSm[j]);
			//		S_MCmin_CF_vmSm[j] = expf(logS_MCmin_vmSm[j]);
			//		S_MCmax_CF_vmSm[j] = expf(logS_MCmax_vmSm[j]);
			//	}
			//}
			//else if (isStrikePriceQuote == 0){
			//	for (long j = 0; j < StockSize; j++){
			//		S_MC_CF_vpSp[j] = expf(logS_MC_vpSp[j])/BasePrice[j] * 100.0f;
			//		S_MCmin_CF_vpSp[j] = expf(logS_MCmin_vpSp[j])/BasePrice[j] * 100.0f;
			//		S_MCmax_CF_vpSp[j] = expf(logS_MCmax_vpSp[j])/BasePrice[j] * 100.0f;

			//		S_MC_CF_vpSm[j] = expf(logS_MC_vpSm[j])/BasePrice[j] * 100.0f;
			//		S_MCmin_CF_vpSm[j] = expf(logS_MCmin_vpSm[j])/BasePrice[j] * 100.0f;
			//		S_MCmax_CF_vpSm[j] = expf(logS_MCmax_vpSm[j])/BasePrice[j] * 100.0f;

			//		S_MC_CF_vmSp[j] = expf(logS_MC_vmSp[j])/BasePrice[j] * 100.0f;
			//		S_MCmin_CF_vmSp[j] = expf(logS_MCmin_vmSp[j])/BasePrice[j] * 100.0f;
			//		S_MCmax_CF_vmSp[j] = expf(logS_MCmax_vmSp[j])/BasePrice[j] * 100.0f;

			//		S_MC_CF_vmSm[j] = expf(logS_MC_vmSm[j])/BasePrice[j] * 100.0f;
			//		S_MCmin_CF_vmSm[j] = expf(logS_MCmin_vmSm[j])/BasePrice[j] * 100.0f;
			//		S_MCmax_CF_vmSm[j] = expf(logS_MCmax_vmSm[j])/BasePrice[j] * 100.0f;
			//	}
			//}
		}
			
		// Price
		for (long j = 0; j < FXSize; j++){
			S_Payoff[0][j] = S_MC_CF[j];
			S_Payoffmin[0][j] = S_MCmin_CF[j];
			S_Payoffmax[0][j] = S_MCmax_CF[j];
		}
		Smin[0] = SMin(S_Payoffmin, FXSize, 0);
		Smax[0] = SMax(S_Payoffmax, FXSize, 0);
		if (price_status == 0){
			if(PayoffCheck(S_Payoff, Smin, Smax, FXSize, i, 0)){					// Checking Redemption
				price_tmp = PayoffCalc(S_Payoff, Smin, Smax, FXSize, i, 0) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
				price_status++;
				result[id].prob = i;
			}
		}

			
		if (SimMode > 0){
			// Delta & Gamma
			for (long j = 0; j < 2 * FXSize; j++){
				for (long k = 0; k < FXSize; k++){
					S_Payoff[j][k] = S_MC_CF[k];
					S_Payoffmin[j][k] = S_MCmin_CF[k];
					S_Payoffmax[j][k] = S_MCmax_CF[k];
				}
				switch (j){
					case 0:
						S_Payoff[0][0] = S_MC_CF_Sp[0]; S_Payoffmin[0][0] = S_MCmin_CF_Sp[0]; S_Payoffmax[0][0] = S_MCmax_CF_Sp[0];
					case 1:
						S_Payoff[1][0] = S_MC_CF_Sm[0]; S_Payoffmin[1][0] = S_MCmin_CF_Sm[0]; S_Payoffmax[1][0] = S_MCmax_CF_Sm[0];
					case 2:
						S_Payoff[2][1] = S_MC_CF_Sp[1]; S_Payoffmin[2][1] = S_MCmin_CF_Sp[1]; S_Payoffmax[2][1] = S_MCmax_CF_Sp[1];
					case 3:
						S_Payoff[3][1] = S_MC_CF_Sm[1]; S_Payoffmin[3][1] = S_MCmin_CF_Sm[1]; S_Payoffmax[3][1] = S_MCmax_CF_Sm[1];
					case 4:
						S_Payoff[4][2] = S_MC_CF_Sp[2]; S_Payoffmin[4][2] = S_MCmin_CF_Sp[2]; S_Payoffmax[4][2] = S_MCmax_CF_Sp[2];
					case 5:
						S_Payoff[5][2] = S_MC_CF_Sm[2]; S_Payoffmin[5][2] = S_MCmin_CF_Sm[2]; S_Payoffmax[5][2] = S_MCmax_CF_Sm[2];
					default:
						break;
				}
			}
			for (long j = 0; j < 2 * FXSize; j++){
				Smin[j] = SMin(S_Payoffmin, FXSize, j);
				Smax[j] = SMax(S_Payoffmax, FXSize, j);
			}
			for (long j = 0; j < 2*FXSize; j++){
				if (delta_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, FXSize, i, j)){					// Checking Redemption
						delta_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, FXSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
						(delta_status[j])++;
					}
				}

				//if (gamma_status[j] == 0){
				//	if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
				//		gamma_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
				//		(gamma_status[j])++;
				//	}
				//}
			}		

			// Vega
			for (long j = 0; j < 2 * FXSize; j++){
				for (long k = 0; k < FXSize; k++){
					S_Payoff[j][k] = S_MC_CF[k];
					S_Payoffmin[j][k] = S_MCmin_CF[k];
					S_Payoffmax[j][k] = S_MCmax_CF[k];
				}
				switch (j){
					case 0:
						S_Payoff[0][0] = S_MC_CF_vp[0]; S_Payoffmin[0][0] = S_MCmin_CF_vp[0]; S_Payoffmax[0][0] = S_MCmax_CF_vp[0];
					case 1:
						S_Payoff[1][0] = S_MC_CF_vm[0]; S_Payoffmin[1][0] = S_MCmin_CF_vm[0]; S_Payoffmax[1][0] = S_MCmax_CF_vm[0];
					case 2:
						S_Payoff[2][1] = S_MC_CF_vp[1]; S_Payoffmin[2][1] = S_MCmin_CF_vp[1]; S_Payoffmax[2][1] = S_MCmax_CF_vp[1];
					case 3:
						S_Payoff[3][1] = S_MC_CF_vm[1]; S_Payoffmin[3][1] = S_MCmin_CF_vm[1]; S_Payoffmax[3][1] = S_MCmax_CF_vm[1];
					case 4:
						S_Payoff[4][2] = S_MC_CF_vp[2]; S_Payoffmin[4][2] = S_MCmin_CF_vp[2]; S_Payoffmax[4][2] = S_MCmax_CF_vp[2];
					case 5:
						S_Payoff[5][2] = S_MC_CF_vm[2]; S_Payoffmin[5][2] = S_MCmin_CF_vm[2]; S_Payoffmax[5][2] = S_MCmax_CF_vm[2];
					default:
						break;
				}
			}
			for (long j = 0; j < 2 * FXSize; j++){
				Smin[j] = SMin(S_Payoffmin, FXSize, j);
				Smax[j] = SMax(S_Payoffmax, FXSize, j);
			}
			for (long j = 0; j < 2 * FXSize; j++){
				if (vega_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, FXSize, i, j)){					// Checking Redemption
						vega_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, FXSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
						(vega_status[j])++;
					}
				}
			}
		}

		if (SimMode > 1){
			// Rho
			for (long j = 0; j < 2 * FXSize; j++){
				for (long k = 0; k < FXSize; k++){
					S_Payoff[j][k] = S_MC_CF[k];
					S_Payoffmin[j][k] = S_MCmin_CF[k];
					S_Payoffmax[j][k] = S_MCmax_CF[k];
				}
				switch (j){
					case 0:
						S_Payoff[0][0] = S_MC_CF_rp[0]; S_Payoffmin[0][0] = S_MCmin_CF_rp[0]; S_Payoffmax[0][0] = S_MCmax_CF_rp[0];
					case 1:
						S_Payoff[1][0] = S_MC_CF_rm[1]; S_Payoffmin[1][0] = S_MCmin_CF_rm[1]; S_Payoffmax[1][0] = S_MCmax_CF_rm[1];
					case 2:
						S_Payoff[2][2] = S_MC_CF_rp[2]; S_Payoffmin[2][2] = S_MCmin_CF_rp[2]; S_Payoffmax[2][2] = S_MCmax_CF_rp[2];
					case 3:
						S_Payoff[0][0] = S_MC_CF_rp[0]; S_Payoffmin[0][0] = S_MCmin_CF_rp[0]; S_Payoffmax[0][0] = S_MCmax_CF_rp[0];
					case 4:
						S_Payoff[1][1] = S_MC_CF_rp[1]; S_Payoffmin[1][1] = S_MCmin_CF_rp[1]; S_Payoffmax[1][1] = S_MCmax_CF_rp[1];
					case 5:
						S_Payoff[2][2] = S_MC_CF_rp[2]; S_Payoffmin[2][2] = S_MCmin_CF_rp[2]; S_Payoffmax[2][2] = S_MCmax_CF_rp[2];
					default:
						break;
				}
			}
			for (long j = 0; j < FXSize; j++){
				Smin[j] = SMin(S_Payoffmin, FXSize, j);
				Smax[j] = SMax(S_Payoffmax, FXSize, j);
			}
			for (long j = 0; j < FXSize; j++){
				if (rho_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, FXSize, i, j)){					// Checking Redemption
						if (j == 0){
							rho_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, FXSize, i, j) * expf(-ytmp*(float)(Schedule[i].T_pay)*dt);
						}
						else{
							rho_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, FXSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
						}
						(rho_status[j])++;
					}
				}
			}

			// Theta
			for (long j = 0; j < FXSize; j++){
				S_Payoff[0][j] = S_MC_CF_tm[j];
				S_Payoffmin[0][j] = S_MCmin_CF_tm[j];
				S_Payoffmax[0][j] = S_MCmax_CF_tm[j];
			}
			for (long j = 0; j < FXSize; j++){
				Smin[j] = SMin(S_Payoffmin, FXSize, j);
				Smax[j] = SMax(S_Payoffmax, FXSize, j);
			}
			if (theta_status < 1){
				if(PayoffCheck(S_Payoff, Smin, Smax, FXSize, i, 0)){					// Checking Redemption
					theta_tmp = PayoffCalc(S_Payoff, Smin, Smax, FXSize, i, 0) * expf(-ytmtm*(float)(Schedule[i].T_pay-1)*dt);
					theta_status++;
				}
			}
		}

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
		//				vanna_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
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
		//				volga_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
		//				(volga_status[j])++;
		//			}
		//		}
		//	}
		//}
	}

	result[id].price = price_tmp;
	if (SimMode > 0){
		for (long i = 0; i < FXSize; i++)
			result[id].delta[i] = (delta_tmp[2*i] - delta_tmp[2*i+1]) / (2.0f * 0.01f * FX[i].S);
		//for (long i = 0; i < FXSize; i++)
			//result[id].gamma[i] = (gamma_tmp[2*i] - 2.0f * price_tmp + gamma_tmp[2*i+1]) / (0.01f * Stock[i].S * 0.01f * Stock[i].S);
		for (long i = 0; i < FXSize; i++)
			result[id].vega[i] = (vega_tmp[2*i] - vega_tmp[2*i+1]) / 2.0f;
	}
	if (SimMode > 1){
		for (long i = 0; i < FXSize; i++)
			result[id].rho[i] = (rho_tmp[i] - price_tmp) / 0.001f;
		//result[id].theta = price_tmp - theta_tmp;
	}
	if (SimMode > 2){
		//for (long i = 0; i < StockSize; i++)
		//	result[id].vanna[i] = ((vanna_tmp[4*i] - vanna_tmp[4*i+1]) - (vanna_tmp[4*i+2] - vanna_tmp[4*i+3]))/ (2.0f * 2.0f * 0.01f * Stock[i].S);
		//for (long i = 0; i < StockSize; i++)
		//	result[id].volga[i] = (volga_tmp[2*i] - 2.0f * price_tmp + volga_tmp[2*i+1]) / (2.0f * 2.0f);
	}

}


// YTM longerp/extrapolation
__device__ float YTMInterp(float t, long YTMType, long YTMSize){
	float r = 0; 
	float t_prior = 0; float r_prior; 
	float YTM_interp;
	long tind = 0;
	if (YTMType == 0)
		r = YTM[0];
	else if (YTMType == 1){
		r_prior = YTM[0];
		while (t > YTMt[tind]){
			r += (r_prior + YTM[tind]) / 2.0 * (YTMt[tind] - t_prior);
			t_prior = YTMt[tind];
			r_prior = YTM[tind];
			tind++;
		}
		YTM_interp = YTM[tind-1] + (YTM[tind] - YTM[tind-1])/(YTMt[tind] - YTMt[tind-1])*(t-YTMt[tind]);
		r += (r_prior + YTM_interp) / 2.0 * (t - t_prior);
		r /= t;
	}
	return r;
}

// Risk-free rate longerp/extrpolation
__device__ float RatedInterp(float t, long fxnum){
	float Rf;
	long tind = 0;

	// Fixed case
	if (FX[fxnum].RatedType == 0){
		Rf = Rated[fxnum*RateTMax];
	}

	// Term-structure case
	else if (FX[fxnum].RatedType == 1){
		while (t > Ratedt[RateTMax*fxnum + tind] && tind < FX[fxnum].RatedSize){
			tind++;
			if (tind == FX[fxnum].RatedSize)	break;
		}
	
		// nearest extrapolation
		if (tind == 0)								Rf = Rated[RateTMax*fxnum];
		else if (tind == FX[fxnum].RatedSize)	Rf = Rated[RateTMax*fxnum + FX[fxnum].RatedSize-1];
		else{
			// linear longerpolation
			Rf = Rated[RateTMax*fxnum + tind-1] + 
				 (Rated[RateTMax*fxnum + tind] - Rated[RateTMax*fxnum + tind-1])/(Ratedt[RateTMax*fxnum + tind] - Ratedt[RateTMax*fxnum + tind-1]) *
				 (t-Ratedt[RateTMax*fxnum + tind-1]);
		}
	}
	return Rf;
}


__device__ float RatefInterp(float t, long fxnum){
	float Rf;
	long tind = 0;

	// Fixed case
	if (FX[fxnum].RatefType == 0){
		Rf = Ratef[fxnum*RateTMax];
	}

	// Term-structure case
	else if (FX[fxnum].RatefType == 1){
		while (t > Rateft[RateTMax*fxnum + tind] && tind < FX[fxnum].RatefSize){
			tind++;
			if (tind == FX[fxnum].RatefSize)	break;
		}
	
		// nearest extrapolation
		if (tind == 0)								Rf = Ratef[RateTMax*fxnum];
		else if (tind == FX[fxnum].RatefSize)		Rf = Ratef[RateTMax*fxnum + FX[fxnum].RatefSize-1];
		else{
			// linear longerpolation
			Rf = Ratef[RateTMax*fxnum + tind-1] + 
				 (Ratef[RateTMax*fxnum + tind] - Ratef[RateTMax*fxnum + tind-1])/(Rateft[RateTMax*fxnum + tind] - Rateft[RateTMax*fxnum + tind-1]) *
				 (t-Rateft[RateTMax*fxnum + tind-1]);
		}
	}
	return Rf;
}


__device__ float VolInterp(float t, float K, long fxnum){
	float v;
	float Vol1, Vol2, Vol11, Vol12, Vol21, Vol22;
	long tind = 0, Kind = 0;

	// Fixed case
	if (FX[fxnum].VolType == 0)
		v = Vol[fxnum*VolTMax*VolKMax];

	// Term structure case
	else if (FX[fxnum].VolType == 1){
		if (t > Volt[VolTMax*fxnum + tind] && tind < FX[fxnum].VolSizet)
			tind++;
		// nearest extrapolation
		if (tind == 0)								v = Vol[VolTMax*fxnum + 0];
		else if (tind == FX[fxnum].VolSizet)		v = Vol[VolTMax*fxnum + FX[fxnum].VolSizet-1];
		else{
			// linear longerpolation
			v = Vol[VolTMax*fxnum + tind-1] + 
				(Vol[VolTMax*fxnum + tind] - Vol[VolTMax*fxnum + tind-1])/(Volt[VolTMax*fxnum + tind] - Volt[VolTMax*fxnum + tind-1]) *
				(t-Volt[VolTMax*fxnum + tind-1]);
		}
	}

	// Surface case
	else if (FX[fxnum].VolType == 2){
		if (t > Volt[VolTMax*fxnum + tind] && tind < FX[fxnum].VolSizet){
				while (t > Volt[VolTMax*fxnum + tind] && tind < FX[fxnum].VolSizet){
					tind++;
					if (tind == FX[fxnum].VolSizet)		break;
			}
	}

	if (K > VolK[VolKMax*fxnum + Kind]){
		while (K > VolK[VolKMax*fxnum + Kind] && Kind < FX[fxnum].VolSizeK){
			Kind++;
			if (Kind == FX[fxnum].VolSizeK)		break;
		}
	}

	if (tind == 0){
		if (Kind == 0)								v = Vol[VolTMax*VolKMax*fxnum + 0];
		else if (Kind == FX[fxnum].VolSizeK)		v = Vol[VolTMax*VolKMax*fxnum + FX[fxnum].VolSizeK - 1];
		else{
			v = Vol[VolTMax*VolKMax*fxnum + Kind-1] + 
			    (Vol[VolTMax*VolKMax*fxnum + Kind] - Vol[VolTMax*VolKMax*fxnum + Kind-1])/(VolK[VolKMax*fxnum + Kind] - VolK[VolKMax*fxnum + Kind-1]) *
			    (K-VolK[VolKMax*fxnum + Kind-1]);
		}
	}
	else if (tind == FX[fxnum].VolSizet){
		if (Kind == 0)								v = Vol[VolTMax*VolKMax*fxnum + VolKMax*(FX[fxnum].VolSizet-1)];
		else if (Kind == FX[fxnum].VolSizeK)		v = Vol[VolTMax*VolKMax*fxnum + VolKMax*(FX[fxnum].VolSizet-1)+FX[fxnum].VolSizeK - 1];
		else{
			v = Vol[VolTMax*VolKMax*fxnum + (VolKMax*(FX[fxnum].VolSizet-1)) + Kind-1] + 
				(Vol[VolTMax*VolKMax*fxnum + (VolKMax*(FX[fxnum].VolSizet-1)) + Kind] - Vol[VolTMax*VolKMax*fxnum + (VolKMax*(FX[fxnum].VolSizet-1)) + Kind-1])/(VolK[VolKMax*fxnum + Kind] - VolK[VolKMax*fxnum + Kind-1]) *
				(K-VolK[VolKMax*fxnum + Kind-1]);
		}
	}
	else{
		if (Kind == 0){
			Vol1 = Vol[VolTMax*VolKMax*fxnum + VolKMax*(tind-1)];
			Vol2 = Vol[VolTMax*VolKMax*fxnum + VolKMax*tind];
			v = Vol1 + (Vol2-Vol1)/(Volt[VolTMax*fxnum + tind] - Volt[VolTMax*fxnum + tind-1]) * (t-Volt[VolTMax*fxnum + tind-1]);
		}
		else if (Kind == FX[fxnum].VolSizeK){
			Vol1 = Vol[VolTMax*VolKMax*fxnum + VolKMax*(tind-1) + FX[fxnum].VolSizeK-1];
			Vol2 = Vol[VolTMax*VolKMax*fxnum + VolKMax*(tind) + FX[fxnum].VolSizeK-1];
			v = Vol1 + (Vol2-Vol1)/(Volt[VolTMax*fxnum + tind] - Volt[VolTMax*fxnum + tind-1]) * (t-Volt[VolTMax*fxnum + tind-1]);
		}
		else{
			Vol11 = Vol[VolTMax*VolKMax*fxnum + VolKMax*(tind-1) + Kind-1];
			Vol12 = Vol[VolTMax*VolKMax*fxnum + VolKMax*(tind-1) + Kind];
			Vol21 = Vol[VolTMax*VolKMax*fxnum + VolKMax*(tind) + Kind-1];
			Vol22 = Vol[VolTMax*VolKMax*fxnum + VolKMax*(tind) + Kind];

			Vol1 = Vol11 + (Vol12-Vol11)/(VolK[VolKMax*fxnum + Kind] - VolK[VolKMax*fxnum + Kind-1]) * (K-VolK[VolKMax*fxnum + Kind-1]);
			Vol2 = Vol21 + (Vol22-Vol21)/(VolK[VolKMax*fxnum + Kind] - VolK[VolKMax*fxnum + Kind-1]) * (K-VolK[VolKMax*fxnum + Kind-1]);

			v = Vol1 + (Vol2-Vol1)/(Volt[VolTMax*fxnum + tind] - Volt[VolTMax*fxnum + tind-1]) * (t-Volt[VolTMax*fxnum + tind-1]);
		}
	}

	}
	return v;
}

// Minimum among stock prices
__device__ float FXMin(float FX_min[][FXSizeMax], long FXSize, long casenum){
	float Min = FX_min[casenum][0];
	for (long i = 1; i < FXSize; i++){
		Min = (FX_min[casenum][i] < Min) ? FX_min[casenum][i] : Min;
	}
	return Min;
}

// Maximum among stock prices
__device__ float FXMax(float FX_max[][FXSizeMax], long FXSize, long casenum){
	float Max = FX_max[casenum][0];
	for (long i = 1; i < FXSize; i++){
		Max = (FX_max[casenum][i] > Max) ? FX_max[casenum][i] : Max;
	}
	return Max;
}

// Reference price
__device__ float RefPriceCalc(float FX[][FXSizeMax], long FXSize, long sched_ind, long casenum){
	float RefPrice = 0;
	switch(Schedule[sched_ind].RefPriceType){
		// Minimum case
		case 0:
			{
				RefPrice = FXMin(FX, FXSize, casenum);
				break;
			}
		// Average case
		case 1:
			{
				for (long i = 0; i < FXSize; i++){					
					RefPrice += FX[casenum][i]/(float)(FXSize);
				}
				break;
			}
		default:
			break;
	}
	return RefPrice;
}

// Checking redemption
__device__ bool PayoffCheck(float FX[][FXSizeMax], float* FX_min, float* FX_max, long FXSize, long sched_ind, long casenum, float accret){
	bool result = false;
	switch(Schedule[sched_ind].BermudanType){
		// Final case
		case 0:
			{
				result = true;
				break;
			}
		// Bermudan case
		case 1:
			{
				switch(Schedule[sched_ind].PayoffType){
				// DIGITCALL case
					case 2:
						{
							if (RefPriceCalc(FX, FXSize, sched_ind, casenum) > Schedule[sched_ind].K)	result = true;
							else																		result = false;
							break;
						}
				// FWDCUMR case
					case 10:
						{
							if (accret > Schedule[sched_ind].RetUpBarrier)	result = true;
							else											result = false;
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
__device__ float PayoffCalc(float S[][FXSizeMax], float* S_min, float* S_max, long StockSize, long sched_ind, long casenum){
	float result = 0;
	switch(Schedule[sched_ind].BermudanType){
		// Final case
		case 0:
			{
				switch(Schedule[sched_ind].PayoffType){
					// PUT
					case 1:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (PayoffPrice > Schedule[sched_ind].K)						result = 100.0f + Schedule[sched_ind].Coupon;
							else if (S_min[casenum] > Schedule[sched_ind].TotalDownBarrier)	result = 100.0f + Schedule[sched_ind].Dummy;
							else															result = SMin(S, StockSize, casenum);
							break;
						}
					// DIGITCALL
					case 2:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (PayoffPrice > Schedule[sched_ind].K)
								result = 100.0f + Schedule[sched_ind].Coupon;
							else if (S_min[casenum] > Schedule[sched_ind].DownBarrier)
								result = 100.0f + Schedule[sched_ind].Dummy;
							else
								result = 100.0f;
							break;
						}
					// PUT_NOKI
					case 3:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (PayoffPrice > Schedule[sched_ind].K)						result = 100.0f + Schedule[sched_ind].Coupon;
							else															result = SMin(S, StockSize, casenum);
							break;
						}
					// KO CALL (coupon acts as a principal value, TotalUpBarrier acts as a barrier)
					case 4:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (S_max[casenum] < Schedule[sched_ind].TotalUpBarrier)
							{
								if (PayoffPrice > Schedule[sched_ind].K)
									result = Schedule[sched_ind].Participation * (PayoffPrice - Schedule[sched_ind].K) + Schedule[sched_ind].Coupon;
								else
									result = Schedule[sched_ind].Coupon;
							}
							else
							{
								result = Schedule[sched_ind].Coupon;
							}
							break;
						}
					// KO PUT (coupon acts as a principal value, TotalUpBarrier acts as a barrier)
					case 6:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (S_max[casenum] < Schedule[sched_ind].TotalUpBarrier)
							{
								if (PayoffPrice < Schedule[sched_ind].K)
									result = Schedule[sched_ind].Participation * (Schedule[sched_ind].K - PayoffPrice) + Schedule[sched_ind].Coupon;
								else
									result = Schedule[sched_ind].Coupon;
							}
							else
							{
								result = Schedule[sched_ind].Coupon;
							}
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
					// DIGITCALL
					case 2:
						{
							result = 100.0f + Schedule[sched_ind].Coupon;
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