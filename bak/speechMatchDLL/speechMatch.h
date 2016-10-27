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

  

int ReadFile(char *wavefile,short* allbuf, int bias);
int ReadFileLength(char* wavefile,int* sampleRate);
void DataScaling(short* data,float* dataScaled);
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
extern int speechMatch(char *file1 ,char *file2, double &a,double &b);

#ifdef __cplusplus  
}  
#endif  

#endif