#ifndef _SPEECHMATCH_H
#define _SPEECHMATCH_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <math.h>
#include <vector>
#include <complex> 
#include <bitset> 
using namespace std;

  
/*
// 获取data中 能量最大的0.5s（或者1/4长度）数据段 的起始位置和长度
*/
int get_window(const short *data, int len_data, int &st_win, int &len_win, int flag);

int ReadFile(const char *wavefile,short* allbuf, int bias, int halfWindow);
int ReadFileLength(const char* wavefile,int* sampleRate);

void DataScaling(short* data,float* dataScaled,int halfWindow);
void InitHamming();
void InitFilt(double *FiltCoe1, double *FiltCoe2, int *Num);
void preemphasis(double* buf, double* result, short FrmLen);
void HammingWindow(double* result,double* data);
void compute_fft(double *data,vector<complex<double> >& vecList);
void CFilt(double *spdata, double *FiltCoe1, double *FiltCoe2, int *Num, double *En,vector<complex<double> >& vecList);
void MFCC(double* En, double* Cep);
void FFT(const unsigned long & ulN, vector<complex<double> >& vecList);

float findLocalMaximum(float* score,int length);
void polyfit(int n,double *x,double *y,int poly_n,double a[]);
void gauss_solve(int n,double A[],double x[],double b[]);




#ifdef __cplusplus  
extern "C" {  
#endif

__declspec(dllexport)  int speechMatch_a(const char *file1 ,const char *file2, double &aa,double &bb);
__declspec(dllexport)  int speechMatch_a_long(const char *file1 ,const char *file2, double &aa,double &bb);
__declspec(dllexport)  int speechMatch(const char *file1 ,const char *file2, 
	const double st_time, const double end_time, double &bb);
__declspec(dllexport)  int speechMatch_large(const char *file1, const char *file2, double &bb);
__declspec(dllexport)  int speechMatch_small(const char *file1, const char *file2, double &bb);
__declspec(dllexport)  int speechMatch_small_v2(const char *file1, const char *file2, double &bb);
__declspec(dllexport)  int speechMatch_conf(const char *file1 ,const char *file2, double &aa,double &bb);

#ifdef __cplusplus  
}  
#endif  

#endif
