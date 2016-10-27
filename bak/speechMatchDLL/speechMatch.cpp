////////////////////
//20150713-20150721
//by wjiang
//@unisound for speechOcean
////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <math.h>
#include <vector>
#include <complex> 
#include <bitset> 
#include "speechMatch.h" 


using namespace std;



#define SAMPRATE 16000
#define HALFWINDOW (SAMPRATE*5)     //16k�����ʵ�ʱ���Ӧ��������ݣ�8k�������Ӧ4��
#define PI 3.1415926536
#define SP_EMPHASIS_FACTOR 0.97f    /* Ԥ���ص�ϵ�� */
#define eps 0.00000000000001

typedef struct _wavhead
{
	char            riff[4];            //"RIFF"
	unsigned long   filelong;           // +8 = File size
	char            wav[8];             //"WAVEfmt "
	unsigned long   t1;                 
	short           tag;
	short           channels;
	unsigned long   samplerate;         
	unsigned long   typesps;            
	unsigned short  psbytes;            
	unsigned short  psbits;             
	char            data[4];            
	unsigned long   sumbytes;           
}WAVEHEAD;
/*
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
int speechMatch(char *file1 ,char *file2, double &a,double &b);
*/

short referenceData[HALFWINDOW*2];//�������
short rawData[HALFWINDOW*2];      //����������
float referenceDataScaled[HALFWINDOW*2];
float rawDataScaled[HALFWINDOW*2];

int sampleRate = 0;
float scaleRate = 0.00008;        //��Ϊ�趨�Ĺ�һ��������ƥ��֮ǰ�Ƚ������ķ��ȱ�СһЩ��

const int FS=16;
const long FrmLen=(FS*20);   //���޸�֡��
double Hamming[FrmLen];
const int FrmNum = HALFWINDOW*2*2/FrmLen-1;

const int FiltNum=40;      //�˲���������һ��40��
const int PCEP=13;         //���õ��Ĺ��ڵ�13��MFCC��ϵ��   

const unsigned long FFTLen=512;    //����FFT�����512�����ݡ�һ��Ϊ����FrmLen��Ϊ2��ָ���η�����С��������Ϊ������ֱ��дΪ�˹̶���ֵ��

double FiltCoe1[FFTLen/2+1];  //��ϵ��
double FiltCoe2[FFTLen/2+1];  //��ϵ��
int    Num[FFTLen/2+1];     //һ����ԣ�ÿ�������������ڵ������˲����У���������õ���صĵڶ����˲���
double dBuffRefence[FrmLen]; 
double dBuffRaw[FrmLen];
double resultReference[FrmLen];  //Ԥ���ؽ��
double resultRaw[FrmLen];  //Ԥ���ؽ��
static double last=0;  //һ���������һ�����ֵ���˵�����Ԥ����
double dataReference[FrmLen];    //�Ӵ���õ�������
double dataRaw[FrmLen];    //�Ӵ���õ�������
vector<complex<double>> vecList;//FFT����֮�������
vector<double>MFCCcoefficient;

#ifdef __cplusplus  
extern "C" {  
#endif  
int speechMatch(char *file1 ,char *file2, double &aa,double &bb)
{

	int ret = 0;
	FILE* fp_debug;	
	fp_debug = fopen("debug.txt","w");

	//referenceSig: ������ݣ�����������Ҫ��������ݽ��ж���
	//rawSig:       ���������ݣ��ó�����ɵĹ����Ǽ���������������ݵ�ʱ���
	//resultFile:   ����ļ������б����ڼ����ؼ�ʱ����ϴ����������������ݵ�ʱ���
    
	int referenceLength = ReadFileLength(file1,&sampleRate);//��ȡ��������ĳ��ȺͲ�����
	if (referenceLength<sampleRate*140)                       //�������ݵĳ���Ҫ����140s���ϣ������ֳ�7�εĻ������ܱ�֤ÿ����������20s�ĳ��ȡ�
	{
		printf("The reference wave file is too short!");
		exit(-2);
	}
	if (sampleRate!=SAMPRATE)
	{
		printf("The sampling rate should be 16kHz!");
		exit(-3);
	}
	//���ڴ�����������������ݳ����ǻ���һ���ģ���˲��ٶԴ��������ݵĳ��Ƚ��п��졣
	int numPart = 7;//numPart=7��ʾ�����ݷֳɵȳ���7��
	
	int bias = 0;
	double positions[6];//numPart-2+1
	double deviations[6];//numPart-2+1
	for (int par=1;par<numPart;par++)//����ԭʼ����һ��Ƚϴ󣬲��ú÷������Ļ��ڴ治����������Ҫ�ֿ���롣
	{                                //���ң���ÿ��part��Ҳ����ÿ��ȫ�����룬����ֻ����HALFWINDOW*2��10���ӣ������ݣ����ڶ��롣
		double Cep_reference_part[(PCEP-1)*FrmNum];//MFCC���
		double Cep_raw_part[(PCEP-1)*FrmNum];//MFCC���

		bias = (int(referenceLength/numPart)*par-HALFWINDOW)*sizeof(short);
		// ��bias����ʼ��ȡ hanming��
		ret = ReadFile(file1,referenceData,bias);	
		if(ret < 0)
		{
			printf("ReadFile:%s error!\n",file1);
			return 0;
		}
		ret = ReadFile(file2,rawData,bias);
		if(ret < 0)
		{
			printf("ReadFile:%s error!\n",file2);
			return 0;
		}
		DataScaling(referenceData,referenceDataScaled);
		DataScaling(rawData,rawDataScaled);
		
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//	% PartI: compute MFCC
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		InitHamming();//��ʼ��������
		InitFilt(FiltCoe1,FiltCoe2,Num); //��ʼ��MEL�˲�ϵ��

		double En[FiltNum+1];         //Ƶ������
		double Cep_reference[PCEP];//MFCC���
		double Cep_raw[PCEP];//MFCC���
		for (int frame=0;frame<FrmNum;frame++) // ÿһ֡
		{
			if (frame==FrmNum-1)
			{
				printf("");
			}
			
			for (int j = 0; j < FrmLen; j++)
			{
				dBuffRefence[j] = (double)referenceDataScaled[frame*FrmLen/2+j];//�õ�һ֡����
				dBuffRaw[j] = (double)rawDataScaled[frame*FrmLen/2+j];
			}

			preemphasis(dBuffRefence,resultReference,FrmLen);//Ԥ���ؽ������result����
			HammingWindow(resultReference,dataReference); //��һ֡���ݼӴ�,����data����
			compute_fft(dataReference,vecList);
			CFilt(dataReference, FiltCoe1, FiltCoe2, Num, En,vecList);
			MFCC(En, Cep_reference);
			vecList.clear();

			preemphasis(dBuffRaw,resultRaw,FrmLen);//Ԥ���ؽ������result����
			HammingWindow(resultRaw,dataRaw); //��һ֡���ݼӴ�,����data����
			compute_fft(dataRaw,vecList);
			CFilt(dataRaw, FiltCoe1, FiltCoe2, Num, En,vecList);
			MFCC(En, Cep_raw);
			vecList.clear();

			for (int tick = 0;tick<PCEP-1;tick++)  // ��һ֡��12��mfccϵ��
			{
				Cep_reference_part[frame*(PCEP-1)+tick] = Cep_reference[tick+1];  // ÿһ֡������������
				Cep_raw_part[frame*(PCEP-1)+tick] = Cep_raw[tick+1];
			}			

		}//end of frame cycle
		int startFrame = 400;
		int numFrameMiddle = 200;
		float vectorMFCC_ref[2400];//numFrameMiddle*(PCEP-1)
		int numFramePrePos = 290;
		float vectorMFCC_raw[2400];
		float score[580];//numFramePrePos*2
		for (int i=startFrame;i<startFrame+numFrameMiddle;i++)  // ȡ 400-600���֡
		{
			for (int j=0;j<PCEP-1;j++)
			{
				vectorMFCC_ref[(i-startFrame)*(PCEP-1)+j] = Cep_reference_part[i*(PCEP-1)+j];
			}
		}
		for (int pole = -numFramePrePos;pole<numFramePrePos;pole++)   // -290~290
		{
			score[pole+numFramePrePos] = 0;
			for (int i=startFrame+pole;i<startFrame+numFrameMiddle+pole;i++) // 
			{
				for (int j=0;j<PCEP-1;j++)
				{
					vectorMFCC_raw[(i-startFrame-pole)*(PCEP-1)+j] = Cep_raw_part[i*(PCEP-1)+j];
				}
			}
			for (int tick=0;tick<numFrameMiddle*(PCEP-1);tick++)
			{
				score[pole+numFramePrePos] += vectorMFCC_ref[tick]*vectorMFCC_raw[tick]/float(numFrameMiddle*(PCEP-1));
			}
		}
		//positions[par-1] = double(bias)/2.0/16000.0/60.0;//��λ�Ƿ�
		positions[par-1] = double(bias)/2.0/16000.0;//��λ����
		float thisDeviation = findLocalMaximum(score,numFramePrePos*2);
		//deviations[par-1] = (double)thisDeviation;
		deviations[par-1] = (double)thisDeviation/100.0 + positions[par-1];		

	}//end of part cycle

	int poly_n=1;
	double a[3];//poly_n+1
	polyfit(numPart-2,positions,deviations,poly_n,a);
	aa=a[1];
	bb=a[0];
	//EMatrix(positions,deviations,numPart-2,poly_n,a);

	fclose(fp_debug);
	return 0;
}
#ifdef __cplusplus  
}  
#endif  


/*
void main(int argc, char* argv[])
{
	FILE* ofp;
	double aa = 0.0;
	double bb = 0.0;	
	
	if(argc!=4)
	{
		printf("Please input arguments: wav1 wav2 out_file\n");
		printf("//wav1: ������ݣ�����������Ҫ��������ݽ��ж���\nwav2:���������ݣ��ó�����ɵĹ����Ǽ���������������ݵ�ʱ���\nwav2:����ļ������б����ڼ����ؼ�ʱ����ϴ����������������ݵ�ʱ���");
		exit(0);
	}	
	
	if ((ofp=fopen(argv[3],"w"))==NULL)
	{
		printf("Can not open output file!\n");
		exit(-1);
	}
    
	int ret = speechMatch(argv[1],argv[2],aa,bb);
	if(ret != 0)
	{
		printf("speechMatch error! result is %d\n",ret);
		exit(0);
	}
	fprintf(ofp,"%.4f ",aa);
	fprintf(ofp,"%.4f ",bb);
	fprintf(ofp,"\n");

	fclose(ofp);

	
}
*/

void gauss_solve(int n,double A[],double x[],double b[])
{
	int i,j,k,r;
	double max;
	for (k=0;k<n-1;k++)
	{
		max=fabs(A[k*n+k]); /*find maxmum*/
		r=k;
		for (i=k+1;i<n-1;i++)
			if (max<fabs(A[i*n+i]))
			{
				max=fabs(A[i*n+i]);
				r=i;
			}
			if (r!=k)
				for (i=0;i<n;i++) /*change array:A[k]&A[r] */
				{
					max=A[k*n+i];
					A[k*n+i]=A[r*n+i];
					A[r*n+i]=max;
				}
				max=b[k]; /*change array:b[k]&b[r] */
				b[k]=b[r];
				b[r]=max;
				for (i=k+1;i<n;i++)
				{
					for (j=k+1;j<n;j++)
						A[i*n+j]-=A[i*n+k]*A[k*n+j]/A[k*n+k];
					b[i]-=A[i*n+k]*b[k]/A[k*n+k];
				}
	}
	for (i=n-1;i>=0;x[i]/=A[i*n+i],i--)
		for (j=i+1,x[i]=b[i];j<n;j++)
			x[i]-=A[i*n+j]*x[j];
}

/*==================polyfit(n,x,y,poly_n,a)===================*/
/*=======���y=a0+a1*x+a2*x^2+����+apoly_n*x^poly_n========*/
/*=====n�����ݸ��� xy������ֵ poly_n�Ƕ���ʽ������======*/
/*===����a0,a1,a2,����a[poly_n]��ϵ����������һ�������=====*/
void polyfit(int n,double x[],double y[],int poly_n,double a[])
{
	int i,j;
	double *tempx,*tempy,*sumxx,*sumxy,*ata;
	tempx=new double[n];
	sumxx=new double[poly_n*2+1];
	tempy=new double[n];
	sumxy=new double[poly_n+1];
	ata=new double[(poly_n+1)*(poly_n+1)];
	for (i=0;i<n;i++)
	{
		tempx[i]=1;
		tempy[i]=y[i];
	}
	for (i=0;i<2*poly_n+1;i++)
		for (sumxx[i]=0,j=0;j<n;j++)
		{
			sumxx[i]+=tempx[j];
			tempx[j]*=x[j];
		}
		for (i=0;i<poly_n+1;i++)
			for (sumxy[i]=0,j=0;j<n;j++)
			{
				sumxy[i]+=tempy[j];
				tempy[j]*=x[j];
			}
			for (i=0;i<poly_n+1;i++)
				for (j=0;j<poly_n+1;j++)
					ata[i*(poly_n+1)+j]=sumxx[i+j];
			gauss_solve(poly_n+1,ata,a,sumxy);
			delete [] tempx;
			tempx=NULL;
			delete [] sumxx;
			sumxx=NULL;
			delete [] tempy;
			tempy=NULL;
			delete [] sumxy;
			sumxy=NULL;
			delete [] ata;
			ata=NULL;

}

float findLocalMaximum(float* score,int length)
{
	float maxiScore = 0;
	float maxiPos = 0;
	for (int i=0;i<length;i++)
	{
		if (score[i]>maxiScore)
		{
			maxiScore = score[i];
			maxiPos = i-290;
		}
	}
	return maxiPos; 
}

/*
����MFCCϵ��
���������*En ---����Ƶ������
*/
void MFCC(double* En, double* Cep)
{
	int idcep, iden;
	//	double Cep[13];
	for(idcep = 0 ; idcep < PCEP ; idcep++)
	{ 
		Cep[idcep] = 0.0f;
		for(iden = 0; iden < FiltNum; iden++)   //��ɢ���ұ任
		{
			if(iden == 0)
			{
				Cep[idcep] = Cep[idcep] + En[iden] * (double)cos(idcep * (iden+0.5f) * PI/(FiltNum)) * 10.0f * sqrt(1/(double)FiltNum);
			}
			else
			{
				Cep[idcep] = Cep[idcep] + En[iden] * (double)cos(idcep * (iden+0.5f) * PI/(FiltNum)) * 10.0f * sqrt(2/(double)FiltNum);
			}

		}
		MFCCcoefficient.push_back(Cep[idcep]);
	}
}

/*
�����˲�����������Ƶ������
���������*spdata  ---Ԥ����֮���һ֡�����ź�
*FiltCoe1---�������˲�����ߵ�ϵ��
*FiltCoe2---�������˲����ұߵ�ϵ��
*Num     ---����ÿ����������һ���˲���

���������*En      ---�������Ƶ������
*/
//������ĳһƵ��������ȫ����������
//CFilt(data, FiltCoe1, FiltCoe2, Num, En,vecList); veclist �� FFT������Ľ��  Num:����ÿ����������һ���˲���
void CFilt(double *spdata, double *FiltCoe1, double *FiltCoe2, int *Num, double *En,vector<complex<double> >& vecList)
{

	double temp=0;
	int id, id1, id2;

	for(id = 0; id < FiltNum ; id++)
	{
		En[id]=0.0F;
	}
	for(id = 0 ; id <= FFTLen/2 ; id++)
	{
		temp = vecList[id].real() * vecList[id].real() + vecList[id].imag() * vecList[id].imag();
		temp=temp / ( (FrmLen/2) * (FrmLen/2) );
		id1 = Num[id];
		if (id1 == 0)
			En[id1] = En[id1] + FiltCoe1[id] * temp;
		if (id1 == FiltNum)
			En[id1-1] = En[id1-1] + FiltCoe2[id] * temp;
		if ((id1 > 0) && (id1 < FiltNum))   
		{
			id2 = id1-1;
			En[id1] = En[id1] + FiltCoe1[id] * temp;
			En[id2] = En[id2] + FiltCoe2[id] * temp;
		}
	}
	for(id = 0 ; id < FiltNum ; id++)
	{
		if (En[id] != 0)
			En[id]=(double)log10(En[id]+eps);
	}
}

void FFT(const unsigned long & ulN, vector<complex<double> >& vecList) 
{ 
	//�õ�ָ�������ָ��ʵ����ָ���˼���FFTʱ�ڲ���ѭ������   
	unsigned long ulPower = 0; //ָ�� 
	unsigned long ulN1 = ulN - 1;   //ulN1=511
	while(ulN1 > 0) 
	{ 
		ulPower++; 
		ulN1 /= 2; 
	} 

	//������ΪFFT�����Ľ��������˳��ģ���Ҫ������������������FFTʵ�ʲ��ּ���֮ǰ�ȵ�����Ҳ�����ڽ��
	//����������ٵ����������������ȵ������ټ���FFTʵ�ʲ���
	bitset<sizeof(unsigned long) * 8> bsIndex; //���������� 
	unsigned long ulIndex; //��ת������ 
	unsigned long ulK; 
	for(unsigned long long p = 0; p < ulN; p++) 
	{ 
		ulIndex = 0; 
		ulK = 1; 
		bsIndex = bitset<sizeof(unsigned long) * 8>(p); 
		for(unsigned long j = 0; j < ulPower; j++) 
		{ 
			ulIndex += bsIndex.test(ulPower - j - 1) ? ulK : 0; 
			ulK *= 2; 
		} 

		if(ulIndex > p)     //ֻ�д���ʱ���ŵ����������ֵ�����ȥ��
		{ 
			complex<double> c = vecList[p]; 
			vecList[p] = vecList[ulIndex]; 
			vecList[ulIndex] = c; 
		} 
	} 

	//������ת���� 
	vector<complex<double> > vecW; 
	for(unsigned long i = 0; i < ulN / 2; i++) 
	{ 
		vecW.push_back(complex<double>(cos(2 * i * PI / ulN) , -1 * sin(2 * i * PI / ulN))); 
	} 

	//����FFT 
	unsigned long ulGroupLength = 1; //�εĳ��� 
	unsigned long ulHalfLength = 0; //�γ��ȵ�һ�� 
	unsigned long ulGroupCount = 0; //�ε����� 
	complex<double> cw; //WH(x) 
	complex<double> c1; //G(x) + WH(x) 
	complex<double> c2; //G(x) - WH(x) 
	for(unsigned long b = 0; b < ulPower; b++) 
	{ 
		ulHalfLength = ulGroupLength; 
		ulGroupLength *= 2; 
		for(unsigned long j = 0; j < ulN; j += ulGroupLength) 
		{ 
			for(unsigned long k = 0; k < ulHalfLength; k++) 
			{ 
				cw = vecW[k * ulN / ulGroupLength] * vecList[j + k + ulHalfLength]; 
				c1 = vecList[j + k] + cw; 
				c2 = vecList[j + k] - cw; 
				vecList[j + k] = c1; 
				vecList[j + k + ulHalfLength] = c2; 
			} 
		} 
	} 
} 

void compute_fft(double *data,vector<complex<double> >& vecList)
{	
	for(int i=0;i<FFTLen;++i)
	{
		if(i<FrmLen)
		{
			complex<double> temp(data[i]);
			vecList.push_back(temp);
		}
		else
		{
			complex<double> temp(0);    //��������FFT���ȴ��ڴ���������Ĳ���������䡣�õ���Ч�����
			vecList.push_back(temp);
		}
	}
	FFT(FFTLen,vecList);
}

//��һ֡���ݼӴ�
void HammingWindow(double* result,double* data)
{
	int i;
	for(i=0;i<FrmLen;i++)
	{
		data[i]=result[i]*Hamming[i];
	}

}

//Ԥ����
void preemphasis(double* buf, double* result, short FrmLen)
{
	int i;
	result[0] = buf[0] - SP_EMPHASIS_FACTOR * last;
	for(i=1;i<FrmLen;i++)
	{
		result[i] = buf[i] - SP_EMPHASIS_FACTOR * buf[i-1];
	}
	last = buf[(FrmLen)/2-1];   //����ÿ���ư�֡
}

/*
�����˲�������
�����������
���������*FiltCoe1---�������˲�����ߵ�ϵ��
*FiltCoe2---�������˲����ұߵ�ϵ��
*Num     ---����ÿ����������һ���˲���
*/
void InitFilt(double *FiltCoe1, double *FiltCoe2, int *Num)
{
	int i,k;
	double Freq;
	double FiltFreq[FiltNum+2]; //40���˲���������42���˲����˵㡣ÿһ���˲��������Ҷ˵�ֱ���ǰһ������һ���˲���������Ƶ�����ڵĵ�
	double BW[FiltNum+1]; //������ÿ�����ڶ˵�֮���Ƶ�ʿ��

	double low = (double)( 400.0 / 3.0 );    /* �˲���������Ƶ�ʣ�����һ���˵�ֵ */
	short lin = 13;    /* 1000Hz��ǰ��13���˲��������Եķֲ��� */
	double lin_spacing = (double)( 200.0 / 3.0 );    /* �����˲������ĵľ���Ϊ66.6Hz */
	short log = FiltNum-13;     /* 1000Hz�Ժ���27���������Էֲ����˲��� */
	double log_spacing = 1.0711703f;    /* �����˲������߿�ȵı�ֵ */

	for ( i=0; i<lin; i++)
	{
		FiltFreq[i] = low + i * lin_spacing;//ǰ13���˲����Ĵ���߽�
	}
	for ( i=lin; i<lin+log+2; i++)
	{
		FiltFreq[i] = FiltFreq[lin-1] * (double)pow( log_spacing, i - lin + 1 );//�����˲����Ĵ���߽�
	}
	for ( i=0; i<FiltNum+1; i++)
	{
		BW[i] = FiltFreq[i+1] - FiltFreq[i];//ʹ���˲����Ĵ���߽�������
	}
	for(i = 0 ; i<= FFTLen/2 ; i++ )
	{
		Num[i] = 0;
	}
	bool bFindFilt = false;
	for(i = 0 ; i <= FFTLen/2 ; i++)
	{
		Freq = FS * 1000.0F * i / (double)(FFTLen);
		bFindFilt = false;
		for(k = 0; k <= FiltNum; k++)
		{  
			if(Freq >= FiltFreq[k] && Freq <= FiltFreq[k+1])
			{
				bFindFilt = true;
				if(k == FiltNum)
				{
					FiltCoe1[i]=0.0F;
				}
				else
				{
					FiltCoe1[i] = (Freq - FiltFreq[k]) / (double)(BW[k]) * 2.0f / (BW[k] + BW[k+1]);
				}
				if(k == 0)
				{  	
					FiltCoe2[i] = 0.0F;
				}
				else
				{	
					FiltCoe2[i] = (FiltFreq[k+1] - Freq) / (double)(BW[k]) * 2.0f / (BW[k] + BW[k-1]);
				}

				Num[i] = k;		//��k==FiltNumʱ����Ϊ��FiltNum���˲�����ʵ�������������ڡ�����ֻ��Ϊ�˼��㷽�㣬�����е�FiltNum���˲������ڡ�
				//����ʵ�Ⲣ��Ӱ����
				break;
			}
		}
		if (!bFindFilt)
		{
			Num[i] = 0;    //��ʱ���õ㲻�����κ��˲�������Ϊ������ϵ����Ϊ0�����Կ��Լٶ�������ĳ���˲�����������Ӱ������������
			//������Ϊ��һ���˲�����
			FiltCoe1[i]=0.0F;
			FiltCoe2[i]=0.0F;	
		}
	}

}
void InitHamming()
{
	double twopi;
	int i;
	twopi=2*PI;
	for( i=0;i<FrmLen;i++)
	{
		Hamming[i]=(double)(0.54-0.46*cos(i*twopi/(double)(FrmLen-1)));
	}
}

void DataScaling(short* data,float* dataScaled)
{
	/*float squreSum = 0;
	for (int i=0;i<HALFWINDOW*2-1;i++)
	{
		squreSum += data[i]*data[i]/float(HALFWINDOW)/2.0;
	}
	for (int i=0;i<HALFWINDOW*2;i++)
	{
		dataScaled[i] = data[i]*(data[i]*data[i]/squreSum);
	}*/

	float maxValue = 0;
	for (int i=0;i<HALFWINDOW*2-1;i++)
	{
		if (data[i]<maxValue)
		{
			maxValue = data[i];
		}
	}
	for (int i=0;i<HALFWINDOW*2;i++)
	{
		dataScaled[i] = data[i]/maxValue;
	}
}

int ReadFile(char *wfile, short* allbuf, int bias)
{
	bool oflag=false;
	FILE *fp=NULL;
	WAVEHEAD head;
	int SAMFREQ=-1;
	int sample_count=0,channel_num=0,readflag=0;
	int numSample = 0;//�����ݳ���
	try
	{
		//�ж������ļ�
		if (strstr(wfile, ".wav")) {
			fp=fopen(wfile, "rb");
			if (fp == NULL) {
				return -1;
			}
			oflag=true;
			fseek(fp,0,SEEK_END);
			sample_count = ftell(fp) - sizeof(WAVEHEAD);
			fseek(fp,0,SEEK_SET);
			fread(&head, 1, sizeof(WAVEHEAD), fp);
			//data
			if(head.data[0]!='d'&&head.data[1]!='a'&&head.data[2]!='t'&&head.data[3]!='a')
			{
				fclose(fp);
				return -1;
			}
			//RIFF
			if(head.riff[0]!='R'&&head.riff[1]!='I'&&head.riff[2]!='F'&&head.riff[3]!='F')
			{
				fclose(fp);
				return -1;
			}
			//"WAVEfmt "
			if(head.wav[0]!='W'&&head.wav[1]!='A'&&head.wav[2]!='V'&&head.wav[3]!='E'&&head.wav[4]!='f'&&head.wav[5]!='m'&&head.wav[6]!='t'&&head.wav[7]!=' ')
			{
				fclose(fp);
				return -1;
			}
			//��λ����
			fseek(fp,(long)(head.t1-16)-4,SEEK_CUR);
			fread(&head.sumbytes,1,sizeof(long),fp);
			//�õ��ֽ���
			sample_count=head.sumbytes;
			if(head.samplerate>48000||head.samplerate<0)
			{
				fclose(fp);
				exit(-1);
			}
			SAMFREQ = head.samplerate;
			channel_num = head.channels;
		}
		//�õ���������n��ͨ���������ͣ���Ϊ16bit��
		sample_count /= sizeof(short);
		if (sample_count % channel_num != 0) {
			fclose(fp);
			return -2;
		}
		//����ռ��ȡ����
		if (bias/2+HALFWINDOW*2<sample_count)
		{
			numSample = HALFWINDOW*2;
		}
		else
		{
			numSample = sample_count-bias/2;
		}
		//allbuf = (short*)malloc(numSample * sizeof(short));
		fseek(fp, bias, SEEK_CUR);
		fread(allbuf, sizeof(short), numSample,fp);
		fclose(fp);
		oflag=false;
	}
	catch(...)
	{
		if(oflag)
			fclose(fp);

		if(allbuf)free(allbuf);
		allbuf=NULL;
		return -1;

	}
	return 0;
}

int ReadFileLength(char *wfile,int* sampleRate)
{
	bool oflag=false;
	FILE *fp=NULL;
	WAVEHEAD head;
	int SAMFREQ=-1;
	int sample_count=0,channel_num=0,readflag=0;
	int numSample = 0;//�����ݳ���
	try
	{
		//�ж������ļ�
		if (strstr(wfile, ".wav")) {
			fp=fopen(wfile, "rb");
			if (fp == NULL) {
				return -1;
			}
			oflag=true;
			fseek(fp,0,SEEK_END);
			sample_count = ftell(fp) - sizeof(WAVEHEAD);
			fseek(fp,0,SEEK_SET);
			fread(&head, 1, sizeof(WAVEHEAD), fp);
			//data
			if(head.data[0]!='d'&&head.data[1]!='a'&&head.data[2]!='t'&&head.data[3]!='a')
			{
				fclose(fp);
				return -1;
			}
			//RIFF
			if(head.riff[0]!='R'&&head.riff[1]!='I'&&head.riff[2]!='F'&&head.riff[3]!='F')
			{
				fclose(fp);
				return -1;
			}
			//"WAVEfmt "
			if(head.wav[0]!='W'&&head.wav[1]!='A'&&head.wav[2]!='V'&&head.wav[3]!='E'&&head.wav[4]!='f'&&head.wav[5]!='m'&&head.wav[6]!='t'&&head.wav[7]!=' ')
			{
				fclose(fp);
				return -1;
			}
			//��λ����
			fseek(fp,(long)(head.t1-16)-4,SEEK_CUR);
			fread(&head.sumbytes,1,sizeof(long),fp);
			//�õ��ֽ���
			sample_count=head.sumbytes;
			if(head.samplerate>48000||head.samplerate<0)
			{
				fclose(fp);
				exit(-1);
			}
			SAMFREQ = head.samplerate;
			channel_num = head.channels;

			*sampleRate = SAMFREQ;
		}
		//�õ���������n��ͨ���������ͣ���Ϊ16bit��
		sample_count /= sizeof(short);
		if (sample_count % channel_num != 0) {
			fclose(fp);
			return -2;
		}
		/*//����ռ��ȡ����
		if (bias+MAX<sample_count)
		{
			numSample = MAX;
		}
		else
		{
			numSample = sample_count-bias;
		}
		allbuf = (short*)malloc(numSample * sizeof(short));
		fread(allbuf, sizeof(short), numSample,fp+bias);
		fclose(fp);
		oflag=false;*/

		return sample_count;
	}
	catch(...)
	{
		if(oflag)
			fclose(fp);

		/*if(allbuf)free(allbuf);
		allbuf=NULL;*/
		return -1;

	}
	return 0;
}

