
#include "stdafx.h"
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

const int SAMPRATE = 16000;
const int FS=16;
const long FrmLen=(FS*20);   //���޸�֡�� 20ms��һ֡���� 16KHz*20ms=320��������
double Hamming[FrmLen];
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



int sampleRate = 0;
float scaleRate = 0.00008;        //��Ϊ�趨�Ĺ�һ��������ƥ��֮ǰ�Ƚ������ķ��ȱ�СһЩ��

const int FiltNum=40;      //�˲���������һ��40��
const int PCEP=13;         //���õ��Ĺ��ڵ�13��MFCC��ϵ��   

const unsigned long FFTLen=512;    //����FFT�����512�����ݡ�һ��Ϊ����FrmLen��Ϊ2��ָ���η�����С��������Ϊ������ֱ��дΪ�˹̶���ֵ��

double FiltCoe1[FFTLen/2+1];  //��ϵ��
double FiltCoe2[FFTLen/2+1];  //��ϵ��
int    Num[FFTLen/2+1];     //һ����ԣ�ÿ�������������ڵ������˲����У���������õ���صĵڶ����˲���
double dBuff_wav_1[FrmLen]; 
double dBuff_wav_2[FrmLen];
double resultReference[FrmLen];  //Ԥ���ؽ��
double resultRaw[FrmLen];  //Ԥ���ؽ��
static double last=0;  //һ���������һ�����ֵ���˵�����Ԥ����
double dataReference[FrmLen];    //�Ӵ���õ�������
double dataRaw[FrmLen];    //�Ӵ���õ�������
vector<complex<double>> vecList;//FFT����֮�������
//vector<double>MFCCcoefficient;


////////////////////////////////////////////////////////////////////////////////////////////

/*
// ��bias����ʼ��ȡ halfWindow ��short�� �������������-1��
*/
int ReadFile(const char *wfile, short* allbuf, int bias, int halfWindow)
{
	bool oflag = false;
	FILE *fp = NULL;
	WAVEHEAD head;
	int SAMFREQ = -1;
	int sample_count = 0, channel_num = 0, readflag = 0;
	int numSample = 0;//�����ݳ���
	try
	{
		//�ж������ļ�
		//if (strstr(wfile, ".wav")) {
		if (true) 
		{
			fp = fopen(wfile, "rb");
			if (fp == NULL) {
				return -2;
			}
			oflag = true;
			fseek(fp, 0, SEEK_END);
			sample_count = ftell(fp) - sizeof(WAVEHEAD);
			fseek(fp, 0, SEEK_SET);
			fread(&head, 1, sizeof(WAVEHEAD), fp);
			//data
			if (head.data[0] != 'd'&&head.data[1] != 'a'&&head.data[2] != 't'&&head.data[3] != 'a')
			{
				fclose(fp);
				return -3;
			}
			//RIFF
			if (head.riff[0] != 'R'&&head.riff[1] != 'I'&&head.riff[2] != 'F'&&head.riff[3] != 'F')
			{
				fclose(fp);
				return -3;
			}
			//"WAVEfmt "
			if (head.wav[0] != 'W'&&head.wav[1] != 'A'&&head.wav[2] != 'V'&&head.wav[3] != 'E'&&head.wav[4] != 'f'&&head.wav[5] != 'm'&&head.wav[6] != 't'&&head.wav[7] != ' ')
			{
				fclose(fp);
				return -3;
			}
			//��λ����
			fseek(fp, (long)(head.t1 - 16) - 4, SEEK_CUR);
			fread(&head.sumbytes, 1, sizeof(long), fp);
			//�õ��ֽ���
			sample_count = head.sumbytes;
			if (head.samplerate>48000 || head.samplerate<0)
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
			return -4;
		}
		// ����ռ��ȡ����
		// ��bias�Ŀ�ʼ��ȡ halfWindow ��short�� �������������-1��
		printf("bias=%d\tsample_count=%d\thalfWindow=%d\n", bias, sample_count, halfWindow);
		if (bias + halfWindow <= sample_count)
		{
			numSample = halfWindow;
		}
		else
		{
			return -5;
		}
		//allbuf = (short*)malloc(numSample * sizeof(short));
		fseek(fp, bias*sizeof(short), SEEK_CUR);
		fread(allbuf, sizeof(short), numSample, fp);

		fclose(fp);
		oflag = false;
	}
	catch (...)
	{
		if (oflag)
		{
			fclose(fp);
		}
		//if(allbuf)free(allbuf);
		//allbuf=NULL;
		return -6;

	}
	return 0;
}


/*
��ȡ�����Ļ�����Ϣ�������ʡ�����
*/
int ReadFileLength(const char *wfile, int* sampleRate)
{
	bool oflag = false;
	FILE *fp = NULL;
	WAVEHEAD head;
	int SAMFREQ = -1;
	int sample_count = 0, channel_num = 0, readflag = 0;
	int numSample = 0;//�����ݳ���
	try
	{
		//�ж������ļ�
		//if (strstr(wfile, ".wav")) {
		if (true) 
		{
			fp = fopen(wfile, "rb");
			if (fp == NULL) {
				printf("read %s err!\n", wfile);
				return -1;
			}
			printf("open file ok!\n");

			oflag = true;
			fseek(fp, 0, SEEK_END);
			sample_count = ftell(fp) - sizeof(WAVEHEAD);
			fseek(fp, 0, SEEK_SET);
			fread(&head, 1, sizeof(WAVEHEAD), fp);
			//data
			if (head.data[0] != 'd'&&head.data[1] != 'a'&&head.data[2] != 't'&&head.data[3] != 'a')
			{
				fclose(fp);
				printf("read data err!\n");
				return -1;
			}
			//RIFF
			if (head.riff[0] != 'R'&&head.riff[1] != 'I'&&head.riff[2] != 'F'&&head.riff[3] != 'F')
			{
				fclose(fp);
				printf("read RIFF err!\n");
				return -1;
			}
			//"WAVEfmt "
			if (head.wav[0] != 'W'&&head.wav[1] != 'A'&&head.wav[2] != 'V'&&head.wav[3] != 'E'&&head.wav[4] != 'f'&&head.wav[5] != 'm'&&head.wav[6] != 't'&&head.wav[7] != ' ')
			{
				fclose(fp);
				printf("read WAVEfmt err!\n");
				return -1;
			}
			//��λ����
			fseek(fp, (long)(head.t1 - 16) - 4, SEEK_CUR);
			fread(&head.sumbytes, 1, sizeof(long), fp);
			//�õ��ֽ���
			sample_count = head.sumbytes;
			if (head.samplerate>48000 || head.samplerate<0)
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
			printf("read channel err!\n");
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

		fclose(fp);
		return sample_count;
	}
	catch (...)
	{
		if (oflag)
			fclose(fp);

		/*if(allbuf)free(allbuf);
		allbuf=NULL;*/
		return -1;

	}

	fclose(fp);
	return 0;
}


/*
	// ��ȡdata��[2/5,3/5]��Χ�� ��������1.0s������1/5���ȣ����ݶ� ����ʼλ�úͳ���
	// 
*/
int get_window(const short *data, int len_data, int &st_win, int &len_win)
{
	int ret = 0;
	short max_short = 30000;
	int len_move = 0.005 * SAMPRATE;  // �ƶ�10ms 
	
	st_win = 0;
	double max_score = 0.0;

	int len_tmp = len_data / 5;

	len_win = (1.0*SAMPRATE < len_tmp) ? 1.0*SAMPRATE : len_tmp;

	// ��data�� 1/4 - 3/4 ��Χ��Ѱ�� 
	int len_14 = 2*len_tmp;
	int len_34 = 3.01*len_tmp;
	for (int ii = len_14; ii < len_34 - len_win; ii += len_move)
	{
		// ���㴰������ 
		double score = 0.0;
		for (int jj = 0; jj < len_win; jj++)
		{
			score += pow(double(data[ii+jj]) / double(max_short), 2);
		}
		if (score > max_score)
		{
			st_win = ii;
			max_score = score;
		}
		
	}


	// ʹ��ԭ�����㷨 
	st_win = len_data / 3.4;
	len_win = len_data / 3 ;
	

	return ret;
}

/*********************************************************************
*
*  //Description: ����С�������루�������ȼ��뵽���������ң�
				// ���뾫��1ms���� 
*
*  Parameters : file1������1·��
*				file2������2·��
*				bb�����b
*  Returns    : ���ش������
*				0����������
*				-1�������ʴ���
*				��������ȡwavʧ��

*********************************************************************/

#ifdef __cplusplus  
extern "C" {
#endif 
	__declspec(dllexport)  int speechMatch_small(const char *file1, const char *file2, 
		double &bb)
	{

		int ret = 0;
		int bak_len_wav1 = 0; // wav1���ܳ��� ���� 
		int bak_len_wav2 = 0; // wav2���ܳ��� ���� 
		int bias_wav1 = 0; // wav1 ��ʼ��ȡ��λ�� 
		int len_wav1 = 0;  // wav1 ��ȡ�ĳ���  �������ڼ��� 
		int bias_wav2 = 0; // wav2 ��ʼ��ȡ�ĳ���  ���⿪ʼѰ�� wav1
		int len_wav2 = 0;  // wav2 ��ȡ�ĳ��� ���������Ѱ�� wav1 

		short *bak_data_wav1 = NULL; // ����wav1 
		short *data_wav1 = NULL; // ���ڼ����wav1
		short *bak_data_wav2 = NULL; // ����wav2 
		short *data_wav2 = NULL; // ���ڼ����wav2

		// ��¼������bak��Ϣ 
		bak_len_wav1 = ReadFileLength(file1, &sampleRate);
		bak_len_wav2 = ReadFileLength(file2, &sampleRate);
		if (sampleRate != SAMPRATE)
		{
			printf("[sampleRate:%d != 16000] !\n", sampleRate);
			return -1;
		}

		// log ���������ʱ��  
		printf("len_wav1=%.4fs\tlen_wav2=%.4fs\n", 
			double(bak_len_wav1) / double(SAMPRATE), 
			double(bak_len_wav2) / double(SAMPRATE));

		// ��ȡ������������ 
		bak_data_wav1 = new short[bak_len_wav1];
		ret = ReadFile(file1, bak_data_wav1, 0, bak_len_wav1);
		bak_data_wav2 = new short[bak_len_wav2];
		ret = ReadFile(file1, bak_data_wav2, 0, bak_len_wav2);
		

		printf("============ ��ʼ��һ��ƥ�� ==============\n");
		// ��ȡwav1��energy�������ݶ�  ����ȡ
		get_window(bak_data_wav1, bak_len_wav1, bias_wav1, len_wav1);
		data_wav1 = new short[len_wav1];	
		// memcpy(data_wav1, bak_data_wav1+bias_wav1, sizeof(short)*len_wav1);
		ret = ReadFile(file1, data_wav1, bias_wav1, len_wav1);
		
		// wav2 ��������Χ 
		bias_wav2 = 0;
		len_wav2 = bak_len_wav2;
		data_wav2 = new short[len_wav2];
		//memcpy(data_wav2, bak_data_wav2, sizeof(short)*len_wav2);	
		ret = ReadFile(file2, data_wav2, bias_wav2, len_wav2);


		printf("wav1: start_time=%.6f s\tlen_time=%.6f s\n", double(bias_wav1) / SAMPRATE, 
			double(len_wav1) / SAMPRATE);
		printf("wav2: start_time=%.6f s\tlen_time=%.6f s\n", double(bias_wav2) / SAMPRATE,
			double(len_wav2) / SAMPRATE);

		////////////////////////////////////////////////////////////////////////
		/// ����len_wav1 len_wav2 data_wav1 data_wav2 ��ƥ��

		// һ֡=20ms	1s=50֡	�� һ֡=16*20��short
		int FrmNum_wav_1 = len_wav1 / FrmLen;
		int FrmNum_wav_2 = len_wav2 / FrmLen;

		int len_window = FrmNum_wav_1;  // ��ȡ�� �Ա�֡ ����
		int fram_move = FrmNum_wav_2 - FrmNum_wav_1; // �����ƶ��ķ�Χ
		float *score = new float[fram_move];
		double *cep_part_1 = new double[(PCEP - 1)*FrmNum_wav_1];
		double *cep_part_2 = new double[(PCEP - 1)*FrmNum_wav_2];

		float *data_scaled_wav1 = new float[len_wav1];
		float *data_scaled_wav2 = new float[len_wav2];

		DataScaling(data_wav1, data_scaled_wav1, len_wav1);
		DataScaling(data_wav2, data_scaled_wav2, len_wav2);


		InitHamming();//��ʼ��������
		InitFilt(FiltCoe1, FiltCoe2, Num); //��ʼ��MEL�˲�ϵ��
		double En[FiltNum + 1];         //Ƶ������
		double Cep_wav_1[PCEP];//MFCC���
		double Cep_wav_2[PCEP];//MFCC���	

		for (int frame = 0; frame<FrmNum_wav_1; frame++) // ÿһ֡
		{
			//�õ�һ֡����
			for (int j = 0; j < FrmLen; j++)
			{
				dBuff_wav_1[j] = (double)data_scaled_wav1[frame*FrmLen + j];

			}
			// ���һ֡�����ݽ��д���
			preemphasis(dBuff_wav_1, resultReference, FrmLen); //Ԥ���ؽ������result����
			HammingWindow(resultReference, dataReference); //��һ֡���ݼӴ�,����data����
			compute_fft(dataReference, vecList);
			CFilt(dataReference, FiltCoe1, FiltCoe2, Num, En, vecList);
			MFCC(En, Cep_wav_1);
			vecList.clear();

			for (int tick = 0; tick<PCEP - 1; tick++)  // ��һ֡��12��mfccϵ��
			{
				// ÿһ֡������������
				cep_part_1[frame*(PCEP - 1) + tick] = Cep_wav_1[tick + 1];
			}
		}

		for (int frame = 0; frame<FrmNum_wav_2; frame++) // ÿһ֡
		{
			//�õ�һ֡����
			for (int j = 0; j < FrmLen; j++)
			{
				dBuff_wav_2[j] = (double)data_scaled_wav2[frame * FrmLen + j];
			}
			// ���һ֡�����ݽ��д���
			preemphasis(dBuff_wav_2, resultRaw, FrmLen);
			HammingWindow(resultRaw, dataRaw);
			compute_fft(dataRaw, vecList);
			CFilt(dataRaw, FiltCoe1, FiltCoe2, Num, En, vecList);
			MFCC(En, Cep_wav_2);
			vecList.clear();

			for (int tick = 0; tick<PCEP - 1; tick++)  // ��һ֡��12��mfccϵ��
			{
				cep_part_2[frame*(PCEP - 1) + tick] = Cep_wav_2[tick + 1];
			}
		}

		// ԭʼ�㷨 
		////printf("�� �����ƶ���֡��=%d\n", fram_move);
		//for (int pole = 0; pole < fram_move; pole++)
		//{
		//	score[pole] = 0;
		//	for (int jj = 0; jj<len_window*(PCEP - 1); jj++)
		//	{
		//		// cos = a*b/|a||b|
		//		float count = cep_part_1[jj] * cep_part_2[pole*(PCEP - 1) + jj] /
		//			float(len_window*(PCEP - 1));
		//		score[pole] += count;
		//
		//	}
		//
		//}
		

		for (int pole = 0; pole < fram_move; pole++)
		{
			score[pole] = 0;
			double ma = 0;
			double mb = 0;
			// ֡��*12 
			for (int jj = 0; jj<len_window*(PCEP - 1); jj++)
			{
				// cos = a*b/|a||b|
				float count = cep_part_1[jj] * cep_part_2[pole*(PCEP - 1) + jj] /
					float(len_window*(PCEP - 1));
				ma += pow(cep_part_1[jj], 2);
				mb += pow(cep_part_2[pole*(PCEP - 1) + jj], 2);
				score[pole] += count;

			}
			score[pole] = score[pole] / (sqrt(ma)*sqrt(mb));

		}

		// ���� len_window �� wav2�еĶ����λ�á�
		float pos = findLocalMaximum(score, fram_move); 		
		bb = (double)(pos*FrmLen + (bias_wav2 - bias_wav1)) / 50.0 / FrmLen;
		printf("bb=%.6f\n",bb);

		//////////////////////////////////////////////////////////////////
		printf("============   ��ʼ��ȷ����....  ============\n");
		// y=ax+b   A������xʱ���  ��ӦB������ax+b ʱ���
		// ��A�����м�λ��x�� ȡ ��������һ��ֵ�� 100ms 
		// ��B������ax+b ����100ms��Χ�ڲ��ҡ�


		int const_pos = 640;  // 40ms   

		// ��Ӧ��B������  x+b �������Ӧ bias_wav1 + b*SAMPRATE 
		int bias_wav2_hhh = double(SAMPRATE) * (bb) + bias_wav1 - const_pos;
		int len_wav2_hhh = len_wav1 + 2 * const_pos;
		short *data_wav2_hhh = new short[len_wav2_hhh];
		ret = ReadFile(file2, data_wav2_hhh, bias_wav2_hhh, len_wav2_hhh);

		// B���������ƶ�  ����
		int fram_move_hhh = len_wav2_hhh - len_wav1;
		float *score_hhh = new float[fram_move_hhh];

		float score_max = 0.0;
		for (int pole = 0; pole < fram_move_hhh; pole++)
		{
			score_hhh[pole] = 0;
			for (int jj = 0; jj<len_wav1; jj++)
			{
				float count = data_wav1[jj] * data_wav2_hhh[pole + jj]
					/ float(len_wav1);
				score_hhh[pole] += count;

			}
		}

		// A������x������  ��Ӧb������: 
		int pos_hhh = findLocalMaximum(score_hhh, fram_move_hhh);
		pos_hhh = bias_wav2_hhh + pos_hhh - bias_wav1;
		printf("pos_hhh=%d\n", pos_hhh);
		bb = double(pos_hhh) / double(SAMPRATE);


		delete[] data_wav2_hhh;
		delete[] score_hhh;

		delete[] data_wav1;
		delete[] data_wav2;
		delete[] data_scaled_wav1;
		delete[] data_scaled_wav2;
		delete[] score;
		delete[] cep_part_1;
		delete[] cep_part_2;

		return 0;


	}
#ifdef __cplusplus  
}
#endif 



/*********************************************************************
*
*  Description: ���������� y=ax+b
*				��С��������ѡȡ�м�3�� ȥ��һ�������в���
*  Parameters : file1������1·��
*				file2������2·��
*				aa�����a
*				bb�����b
*  Returns    : ���ش������
*				0����������
*				-1�������ʴ���
*				1��a != 1
*				��������ȡwavʧ��

*********************************************************************/

#ifdef __cplusplus  
extern "C" {  
#endif  
__declspec(dllexport)  int speechMatch_a(const char *file1, const char *file2, double &aa, double &bb)
{
	int len_for_comp = 900;
	int ret = 0;
	const int numPart = 4;
	FILE* fp_debug;		
	fp_debug = fopen("debug.txt","a+");
	fprintf(fp_debug,"\nwav1:%s\nwav2:%s\n",file1,file2);
	fflush(fp_debug);
	const int max_pos_frame = 300*SAMPRATE; // wav1��wav2�����ƫ��
	fprintf(fp_debug,"malloc mem\n");
	fflush(fp_debug);
	double *positions = new double[numPart-1];
	double *deviations = new double[numPart-1];
    

	// ��������
	fprintf(fp_debug,"read_file_length:%s\n",file1);
	fflush(fp_debug);

	int len_wav_1_old = ReadFileLength(file1, &sampleRate);
	int len_wav_2_old = ReadFileLength(file2, &sampleRate);
	
	const char *file_a = file1;
	const char *file_b = file2;
	int flag_rev = 0;
	// ����file1��file2Ϊ��С��
	if (len_wav_1_old > len_wav_2_old)
	{
		flag_rev = 1;
		file_a = file2;
		file_b = file1;
		int temp = len_wav_1_old;
		len_wav_1_old = len_wav_2_old;
		len_wav_2_old = temp;
	}



	fprintf(fp_debug,"file_a_len=%d��\n",len_wav_1_old/sampleRate);
	fprintf(fp_debug,"file_b_len=%d��\n",len_wav_2_old/sampleRate);
	fflush(fp_debug);
	
	if (sampleRate != SAMPRATE ) 
	{
		fprintf(fp_debug,"[sampleRate:%d != 16000] !\n",sampleRate);
		fflush(fp_debug);
		printf("[sampleRate:%d != 16000] !\n", sampleRate);	
		fclose(fp_debug);
		return -1;
	}


	for (int par=1;par<numPart;par++)
	{
		fprintf(fp_debug,"process %d ...\n",par);
		fflush(fp_debug);
		printf("process %d ...\n",par);
		int len_wav_1 = len_wav_1_old;
		int len_wav_2 = len_wav_2_old;

		double time_wav_1 = double(len_wav_1)/double(SAMPRATE);
		double time_wav_2 = double(len_wav_2)/double(SAMPRATE);

		////////////////////////////////////////////////////////////////////////////////////////
		// �����ȡ wav_1 �Ŀ�ʼ�ͳ���
		int bias_wav_1 = int(time_wav_1 * double(SAMPRATE) / numPart) * par;

		time_wav_1 = time_wav_1 / (numPart * 2) > len_for_comp ? len_for_comp : time_wav_1 / (numPart * 2);
		len_wav_1 = int(time_wav_1*SAMPRATE);
		short *data_wav1 = new short[len_wav_1];
		ret = ReadFile(file_a, data_wav1, bias_wav_1, len_wav_1);	
		if(ret < 0)
		{
			fprintf(fp_debug,"ReadFile:%s error! ret=%d\n",file_a,ret);
			fflush(fp_debug);
			printf("ReadFile:%s error!\n",file_a);
			fclose(fp_debug);
			return ret;
		}
	
		// �����ȡ wav_2 �Ŀ�ʼ�ͳ���
		int bias_wav_2 = 0;
		int temp_len = len_wav_1 + 2*max_pos_frame;  // ��Զȡ��ô�� ����С��
		if(bias_wav_1 > max_pos_frame)
		{
			bias_wav_2 = bias_wav_1 - max_pos_frame; // ҪôΪ0 ҪôΪ��
		}

		if(bias_wav_2 + temp_len > len_wav_2) // �������� ��ô ȡfile_bȫ��
		{
			bias_wav_2 = 0;
			len_wav_2 = len_wav_2 -  bias_wav_2 -2000;			 
		}
		else
		{
			len_wav_2 = temp_len;
		}


		short *data_wav2 = new short[len_wav_2]; 
		ret = ReadFile(file_b, data_wav2, bias_wav_2, len_wav_2);
		if(ret < 0)
		{
			fprintf(fp_debug,"ReadFile:%s error! ret=%d\n",file_b,ret);
			fflush(fp_debug);
			printf("ReadFile:%s error!\n",file_b);
			fclose(fp_debug);
			return ret;
		}
		fprintf(fp_debug,"wav_time[%f:%f]��\n",double(bias_wav_1)/SAMPRATE, double(len_wav_1)/SAMPRATE);
		fprintf(fp_debug,"wav_time[%f:%f]��\n",double(bias_wav_2)/SAMPRATE, double(len_wav_2)/SAMPRATE);
		fflush(fp_debug);
		printf("wav_time[%f:%f]��\n",double(bias_wav_1)/SAMPRATE, double(len_wav_1)/SAMPRATE);
		printf("wav_time[%f:%f]��\n",double(bias_wav_2)/SAMPRATE, double(len_wav_2)/SAMPRATE);
		/////////////////////////////////////////////////////////////////////////////////////////

		// һ֡=20ms	1s=50֡	�� һ֡=16*20��short
		int FrmNum_wav_1 = len_wav_1/FrmLen;
		int FrmNum_wav_2 = len_wav_2/FrmLen;

		int len_window = FrmNum_wav_1;  // ��ȡ�� �Ա�֡ ����
		int fram_move = FrmNum_wav_2 - FrmNum_wav_1; // �����ƶ��ķ�Χ
		float *score = new float[fram_move];
		double *cep_part_1 = new double[(PCEP-1)*FrmNum_wav_1];
		double *cep_part_2 = new double[(PCEP-1)*FrmNum_wav_2];	

		float *data_scaled_wav1 = new float[len_wav_1];
		float *data_scaled_wav2 = new float[len_wav_2];
	
		DataScaling(data_wav1, data_scaled_wav1, len_wav_1);
		DataScaling(data_wav2, data_scaled_wav2, len_wav_2);
		
	
		InitHamming();//��ʼ��������
		InitFilt(FiltCoe1, FiltCoe2, Num); //��ʼ��MEL�˲�ϵ��
		double En[FiltNum+1];         //Ƶ������
		double Cep_wav_1[PCEP];//MFCC���
		double Cep_wav_2[PCEP];//MFCC���	
	
		for (int frame=0; frame<FrmNum_wav_1; frame++) // ÿһ֡
		{		
			//�õ�һ֡����
			for (int j = 0; j < FrmLen; j++)
			{
				dBuff_wav_1[j] = (double)data_scaled_wav1[frame*FrmLen+j];

			}
			// ���һ֡�����ݽ��д���
			preemphasis(dBuff_wav_1, resultReference, FrmLen); //Ԥ���ؽ������result����
			HammingWindow(resultReference, dataReference); //��һ֡���ݼӴ�,����data����
			compute_fft(dataReference, vecList);
			CFilt(dataReference, FiltCoe1, FiltCoe2, Num, En,vecList);
			MFCC(En, Cep_wav_1);
			vecList.clear();

			for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
			{
				// ÿһ֡������������
				cep_part_1[frame*(PCEP-1)+tick] = Cep_wav_1[tick+1];  
			}			
		}

		for (int frame=0; frame<FrmNum_wav_2; frame++) // ÿһ֡
		{		
			//�õ�һ֡����
			for (int j = 0; j < FrmLen; j++)
			{
				dBuff_wav_2[j] = (double)data_scaled_wav2[frame * FrmLen + j];
			}
			// ���һ֡�����ݽ��д���
			preemphasis(dBuff_wav_2, resultRaw, FrmLen);
			HammingWindow(resultRaw, dataRaw); 
			compute_fft(dataRaw, vecList);
			CFilt(dataRaw, FiltCoe1, FiltCoe2, Num, En,vecList);
			MFCC(En, Cep_wav_2);
			vecList.clear();

			for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
			{
				cep_part_2[frame*(PCEP-1)+tick] = Cep_wav_2[tick+1];
			}			
		}

		//printf("�� �����ƶ���֡��=%d\n", fram_move);
		for (int pole = 0; pole < fram_move; pole++)  
		{
			score[pole] = 0;
			for (int jj=0; jj<len_window*(PCEP-1); jj++)
			{
				float count = cep_part_1[jj] * cep_part_2[pole*(PCEP-1)+jj] / float(len_window*(PCEP-1));
				score[pole] += count;
					
			}		
		
		}

		float pos = findLocalMaximum(score, fram_move); // ���� len_window �� wav2�еĶ����λ�á�	
		bb = (double)(pos*FrmLen + (bias_wav_2 - bias_wav_1))/50.0/FrmLen;	

		positions[par-1] = double(bias_wav_1)/50.0/FrmLen;//��λ����
		deviations[par-1] = bb + positions[par-1];	

		if(data_wav1 != NULL){delete [] data_wav1;data_wav1 = NULL; }
		if(data_wav2 != NULL){delete [] data_wav2;data_wav2 = NULL; }      
		if(data_scaled_wav1 != NULL){delete [] data_scaled_wav1; data_scaled_wav1 = NULL; }
		if(data_scaled_wav2 != NULL){delete [] data_scaled_wav2; data_scaled_wav2 = NULL; }
		if(score != NULL){delete [] score;score =NULL;}
		if(cep_part_1 != NULL){delete [] cep_part_1;cep_part_1=NULL;}
		if(cep_part_2 != NULL){delete [] cep_part_2;cep_part_2=NULL;}
		
	
	}

	double dengcha = 0.0;
	double bb_temp = 0.0; // ��ֵ
	double bb_b0 = 0.0; // ����� 
	for(int ii=0; ii<numPart-1; ii++)
	{
		if(ii==0){dengcha -= (deviations[ii] - positions[ii]);}
		if(ii==numPart-2){dengcha += (deviations[ii] - positions[ii]);}
		bb_temp += (deviations[ii] - positions[ii]);
		fprintf(fp_debug,"[%.4f:%.4f]\t%.4f\n",
			positions[ii],deviations[ii],deviations[ii]-positions[ii]);  
	}
	fflush(fp_debug);
	bb_b0 = deviations[0] - positions[0] - dengcha/double(numPart-2);
	bb_temp = bb_temp/double(numPart-1);

	int poly_n=1;
	double a[3];//poly_n+1
	polyfit(numPart-2,positions,deviations,poly_n,a);
	aa=a[1];

	double b_st = deviations[0] - positions[0];
	
	bb = a[0]; 
	// �任
	if(flag_rev == 1)
	{
		aa = 1.0/aa;
		bb = 0.000 - bb/aa;
	}

	fprintf(fp_debug,"aб��:%.4f\n",aa); 
	fprintf(fp_debug,"b��ֵ:%.4f\nb���:%.4f\nb�����:%.4f\nb��ʼ��:%.4f\n��ʹ��:%.4f\n",
		bb_temp,a[0],bb_b0,b_st,bb); 
	

	if(positions != NULL)
	{
		delete [] positions;
		positions = NULL;
	}
	if(deviations != NULL)
	{
		delete [] deviations;
		deviations = NULL;
	}
	if(fp_debug != NULL)
	{
		fclose(fp_debug);
		fp_debug = NULL;
	}
	if(aa < 1.01 && aa > 0.99)
	{
		return 0;
	}
	else
	{
		return 1;
	}
	
}
#ifdef __cplusplus  
}  
#endif  


/*********************************************************************
*
*  Description: ���������� y=ax+b
*				���������룺y=ax+b  ��Ӧ�����������Ȳ��Ƚϴ��  ����ʱ��Ƚϳ���
*  Parameters : file1������1·��
*				file2������2·��
*				aa�����a
*				bb�����b
*  Returns    : ���ش������
*				0����������
*				-1�������ʴ���
*				1��a != 1
*				��������ȡwavʧ��

*********************************************************************/

#ifdef __cplusplus  
extern "C" {  
#endif  
__declspec(dllexport)  int speechMatch_a_long(const char *file1, const char *file2, double &aa, double &bb)
{
	int ret = 0;
	const int numPart = 4;
	FILE* fp_debug;		
	fp_debug = fopen("debug.txt","a+");
	fprintf(fp_debug,"\nwav1:%s\nwav2:%s\n",file1,file2);
	fflush(fp_debug);
	
	fprintf(fp_debug,"malloc mem\n");
	fflush(fp_debug);
	double *positions = new double[numPart-1];
	double *deviations = new double[numPart-1];
    

	// ��������
	fprintf(fp_debug,"read_file_length:%s\n",file1);
	fflush(fp_debug);

	int len_wav_1_old = ReadFileLength(file1, &sampleRate);
	int len_wav_2_old = ReadFileLength(file2, &sampleRate);
	
	const char *file_a = file1;
	const char *file_b = file2;
	int flag_rev = 0;
	// ����file1��file2Ϊ��С��
	if (len_wav_1_old > len_wav_2_old)
	{
		flag_rev = 1;
		file_a = file2;
		file_b = file1;
		int temp = len_wav_1_old;
		len_wav_1_old = len_wav_2_old;
		len_wav_2_old = temp;
	}

	fprintf(fp_debug,"file_a_len=%d��\n",len_wav_1_old/sampleRate);
	fprintf(fp_debug,"file_b_len=%d��\n",len_wav_2_old/sampleRate);
	fflush(fp_debug);
	
	if (sampleRate != SAMPRATE ) 
	{
		fprintf(fp_debug,"[sampleRate:%d != 16000] !\n",sampleRate);
		fflush(fp_debug);
		printf("[sampleRate:%d != 16000] !\n", sampleRate);	
		fclose(fp_debug);
		return -1;
	}


	for (int par=1; par<numPart; par++)
	{
		fprintf(fp_debug,"process %d ...\n",par);
		fflush(fp_debug);
		printf("process %d ...\n",par);
		int len_wav_1 = len_wav_1_old;
		int len_wav_2 = len_wav_2_old;

		double time_wav_1 = double(len_wav_1)/double(SAMPRATE);
		double time_wav_2 = double(len_wav_2)/double(SAMPRATE);

		////////////////////////////////////////////////////////////////////////////////////////
		// �����ȡ wav_1 �Ŀ�ʼ�ͳ���
		int bias_wav_1 = int(time_wav_1 * double(SAMPRATE) / numPart) * par;

		time_wav_1 = time_wav_1/(numPart*2) > 300?300:time_wav_1/(numPart*2); 
		len_wav_1 = int(time_wav_1*SAMPRATE);
		short *data_wav1 = new short[len_wav_1];
		ret = ReadFile(file_a, data_wav1, bias_wav_1, len_wav_1);	
		if(ret < 0)
		{
			fprintf(fp_debug,"ReadFile:%s error! ret=%d\n",file_a,ret);
			fflush(fp_debug);
			printf("ReadFile:%s error!\n",file_a);
			fclose(fp_debug);
			return ret;
		}
	
		// �����ȡ wav_2 �Ŀ�ʼ�ͳ���
		int bias_wav_2 = 0;
		len_wav_2 = len_wav_2 -160000;

		short *data_wav2 = new short[len_wav_2]; 
		ret = ReadFile(file_b, data_wav2, bias_wav_2, len_wav_2);
		if(ret < 0)
		{
			fprintf(fp_debug,"ReadFile:%s error! ret=%d\n",file_b,ret);
			fflush(fp_debug);
			printf("ReadFile:%s error!\n",file_b);
			fclose(fp_debug);
			return ret;
		}
		fprintf(fp_debug,"wav_time[%f:%f]��\n",double(bias_wav_1)/SAMPRATE, double(len_wav_1)/SAMPRATE);
		fprintf(fp_debug,"wav_time[%f:%f]��\n",double(bias_wav_2)/SAMPRATE, double(len_wav_2)/SAMPRATE);
		fflush(fp_debug);
		printf("wav_time[%f:%f]��\n",double(bias_wav_1)/SAMPRATE, double(len_wav_1)/SAMPRATE);
		printf("wav_time[%f:%f]��\n",double(bias_wav_2)/SAMPRATE, double(len_wav_2)/SAMPRATE);
		/////////////////////////////////////////////////////////////////////////////////////////

		// һ֡=20ms	1s=50֡	�� һ֡=16*20��short
		int FrmNum_wav_1 = len_wav_1/FrmLen;
		int FrmNum_wav_2 = len_wav_2/FrmLen;

		int len_window = FrmNum_wav_1;  // ��ȡ�� �Ա�֡ ����
		int fram_move = FrmNum_wav_2 - FrmNum_wav_1; // �����ƶ��ķ�Χ
		float *score = new float[fram_move];
		double *cep_part_1 = new double[(PCEP-1)*FrmNum_wav_1];
		double *cep_part_2 = new double[(PCEP-1)*FrmNum_wav_2];	

		float *data_scaled_wav1 = new float[len_wav_1];
		float *data_scaled_wav2 = new float[len_wav_2];
	
		DataScaling(data_wav1, data_scaled_wav1, len_wav_1);
		DataScaling(data_wav2, data_scaled_wav2, len_wav_2);
		
	
		InitHamming();//��ʼ��������
		InitFilt(FiltCoe1, FiltCoe2, Num); //��ʼ��MEL�˲�ϵ��
		double En[FiltNum+1];         //Ƶ������
		double Cep_wav_1[PCEP];//MFCC���
		double Cep_wav_2[PCEP];//MFCC���	
	
		for (int frame=0; frame<FrmNum_wav_1; frame++) // ÿһ֡
		{		
			//�õ�һ֡����
			for (int j = 0; j < FrmLen; j++)
			{
				dBuff_wav_1[j] = (double)data_scaled_wav1[frame*FrmLen+j];

			}
			// ���һ֡�����ݽ��д���
			preemphasis(dBuff_wav_1, resultReference, FrmLen); //Ԥ���ؽ������result����
			HammingWindow(resultReference, dataReference); //��һ֡���ݼӴ�,����data����
			compute_fft(dataReference, vecList);
			CFilt(dataReference, FiltCoe1, FiltCoe2, Num, En,vecList);
			MFCC(En, Cep_wav_1);
			vecList.clear();

			for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
			{
				// ÿһ֡������������
				cep_part_1[frame*(PCEP-1)+tick] = Cep_wav_1[tick+1];  
			}			
		}

		for (int frame=0; frame<FrmNum_wav_2; frame++) // ÿһ֡
		{		
			//�õ�һ֡����
			for (int j = 0; j < FrmLen; j++)
			{
				dBuff_wav_2[j] = (double)data_scaled_wav2[frame * FrmLen + j];
			}
			// ���һ֡�����ݽ��д���
			preemphasis(dBuff_wav_2, resultRaw, FrmLen);
			HammingWindow(resultRaw, dataRaw); 
			compute_fft(dataRaw, vecList);
			CFilt(dataRaw, FiltCoe1, FiltCoe2, Num, En,vecList);
			MFCC(En, Cep_wav_2);
			vecList.clear();

			for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
			{
				cep_part_2[frame*(PCEP-1)+tick] = Cep_wav_2[tick+1];
			}			
		}

		//printf("�� �����ƶ���֡��=%d\n", fram_move);
		for (int pole = 0; pole < fram_move; pole++)  
		{
			score[pole] = 0;
			for (int jj=0; jj<len_window*(PCEP-1); jj++)
			{
				float count = cep_part_1[jj] * cep_part_2[pole*(PCEP-1)+jj] / float(len_window*(PCEP-1));
				score[pole] += count;
					
			}		
		
		}

		float pos = findLocalMaximum(score, fram_move); // ���� len_window �� wav2�еĶ����λ�á�	
		bb = (double)(pos*FrmLen + (bias_wav_2 - bias_wav_1))/50.0/FrmLen;	

		positions[par-1] = double(bias_wav_1)/50.0/FrmLen;//��λ����
		deviations[par-1] = bb + positions[par-1];	

		if(data_wav1 != NULL){delete [] data_wav1;data_wav1 = NULL; }
		if(data_wav2 != NULL){delete [] data_wav2;data_wav2 = NULL; }      
		if(data_scaled_wav1 != NULL){delete [] data_scaled_wav1; data_scaled_wav1 = NULL; }
		if(data_scaled_wav2 != NULL){delete [] data_scaled_wav2; data_scaled_wav2 = NULL; }
		if(score != NULL){delete [] score;score =NULL;}
		if(cep_part_1 != NULL){delete [] cep_part_1;cep_part_1=NULL;}
		if(cep_part_2 != NULL){delete [] cep_part_2;cep_part_2=NULL;}
		
	
	}

	double dengcha = 0.0;
	double bb_temp = 0.0; // ��ֵ
	double bb_b0 = 0.0; // ����� 
	for(int ii=0; ii<numPart-1; ii++)
	{
		if(ii==0){dengcha -= (deviations[ii] - positions[ii]);}
		if(ii==numPart-2){dengcha += (deviations[ii] - positions[ii]);}
		bb_temp += (deviations[ii] - positions[ii]);
		fprintf(fp_debug,"[%.4f:%.4f]\t%.4f\n",
			positions[ii],deviations[ii],deviations[ii]-positions[ii]);  
	}
	fflush(fp_debug);
	bb_b0 = deviations[0] - positions[0] - dengcha/double(numPart-2);
	bb_temp = bb_temp/double(numPart-1);

	int poly_n=1;
	double a[3];//poly_n+1
	polyfit(numPart-2,positions,deviations,poly_n,a);
	aa=a[1];

	double b_st = deviations[0] - positions[0];
	
	bb = a[0]; 
	// �任
	if(flag_rev == 1)
	{
		aa = 1.0/aa;
		bb = 0.000 - bb/aa;
	}

	fprintf(fp_debug,"aб��:%.4f\n",aa); 
	fprintf(fp_debug,"b��ֵ:%.4f\nb���:%.4f\nb�����:%.4f\nb��ʼ��:%.4f\n��ʹ��:%.4f\n",
		bb_temp,a[0],bb_b0,b_st,bb); 
	

	if(positions != NULL)
	{
		delete [] positions;
		positions = NULL;
	}
	if(deviations != NULL)
	{
		delete [] deviations;
		deviations = NULL;
	}
	if(fp_debug != NULL)
	{
		fclose(fp_debug);
		fp_debug = NULL;
	}
	if(aa < 1.001 && aa > 0.999)
	{
		return 0;
	}
	else
	{
		return 1;
	}
	
}
#ifdef __cplusplus  
}  
#endif 


 

/*
	�������������� 
*/
#ifdef __cplusplus  
extern "C" {  
#endif  
__declspec(dllexport)  int speechMatch_large(const char *file1, const char *file2, double &bb)
{  

	int ret = 0;
	const int max_pos_frame = 180*SAMPRATE; // wav1��wav2�����ƫ�Ʋ�����200s
    
	//��ȡ��������ĳ��ȺͲ����� Ȼ��short����
	//printf("read wav_1:%s\nread wav_2:%s\n", file1, file2);
	int len_wav_1 = ReadFileLength(file1, &sampleRate);
	int len_wav_2 = ReadFileLength(file2, &sampleRate);
	if (sampleRate != SAMPRATE ) 
	{
		printf("[sampleRate:%d != 16000] !\n", sampleRate);	
		return -1;
	}

	double time_wav_1 = double(len_wav_1)/double(SAMPRATE);
	double time_wav_2 = double(len_wav_2)/double(SAMPRATE);
	printf("wav_1=%fs\twav_2=%fs\n",time_wav_1,time_wav_2);

	if(time_wav_1 < 100 || time_wav_2 < 100)
	{
		printf("wav_length < 100s!\n");
		return -100;
	}

	int bias_wav_1 = int(time_wav_1 * double(SAMPRATE) / 3.3);   // �ӵ�bias_wav_1 ֡��ʼ��ȡ wav1�����ݶΣ� 
	time_wav_1 = time_wav_1/5 > 600?600:time_wav_1/5; 
	//time_wav_1 = 60.0;

	len_wav_1 = int(time_wav_1*SAMPRATE);
	short *data_wav1 = new short[len_wav_1];
	ret = ReadFile(file1, data_wav1, bias_wav_1, len_wav_1);	
	if(ret < 0)
	{
		printf("ReadFile:%s error!\n",file1);
		return -2;
	}
	
	
	int bias_wav_2 = 0;
	if(bias_wav_1 > max_pos_frame)
	{
		bias_wav_2 = bias_wav_1 - max_pos_frame;
	}
	int temp_len = len_wav_1 + 2*max_pos_frame;
	if(bias_wav_2 + temp_len < len_wav_2)
	{
		len_wav_2 = temp_len;
	}
	else
	{
		len_wav_2 -=  bias_wav_2 -250;
	}

	short *data_wav2 = new short[len_wav_2]; 
	ret = ReadFile(file2, data_wav2, bias_wav_2, len_wav_2);
	if(ret < 0)
	{
		printf("ReadFile:%s error!\n",file2);
		return -2;
	}
	printf("wav_time[%f:%f]��\n",double(bias_wav_1)/SAMPRATE, double(len_wav_1)/SAMPRATE);
	printf("wav_time[%f:%f]��\n",double(bias_wav_2)/SAMPRATE, double(len_wav_2)/SAMPRATE);
	

	// һ֡=20ms	1s=50֡	�� һ֡=16*20��short
	int FrmNum_wav_1 = len_wav_1/FrmLen;
	int FrmNum_wav_2 = len_wav_2/FrmLen;

	int len_window = FrmNum_wav_1;  // ��ȡ�� �Ա�֡ ����
	int fram_move = FrmNum_wav_2 - FrmNum_wav_1; // �����ƶ��ķ�Χ
	float *score = new float[fram_move];
	double *cep_part_1 = new double[(PCEP-1)*FrmNum_wav_1];
	double *cep_part_2 = new double[(PCEP-1)*FrmNum_wav_2];	

	float *data_scaled_wav1 = new float[len_wav_1];
	float *data_scaled_wav2 = new float[len_wav_2];
	
	DataScaling(data_wav1, data_scaled_wav1, len_wav_1);
	DataScaling(data_wav2, data_scaled_wav2, len_wav_2);
		
	
	InitHamming();//��ʼ��������
	InitFilt(FiltCoe1, FiltCoe2, Num); //��ʼ��MEL�˲�ϵ��
	double En[FiltNum+1];         //Ƶ������
	double Cep_wav_1[PCEP];//MFCC���
	double Cep_wav_2[PCEP];//MFCC���	
	
	for (int frame=0; frame<FrmNum_wav_1; frame++) // ÿһ֡
	{		
		//�õ�һ֡����
		for (int j = 0; j < FrmLen; j++)
		{
			dBuff_wav_1[j] = (double)data_scaled_wav1[frame*FrmLen+j];

		}
		// ���һ֡�����ݽ��д���
		preemphasis(dBuff_wav_1, resultReference, FrmLen); //Ԥ���ؽ������result����
		HammingWindow(resultReference, dataReference); //��һ֡���ݼӴ�,����data����
		compute_fft(dataReference, vecList);
		CFilt(dataReference, FiltCoe1, FiltCoe2, Num, En,vecList);
		MFCC(En, Cep_wav_1);
		vecList.clear();

		for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
		{
			// ÿһ֡������������
			cep_part_1[frame*(PCEP-1)+tick] = Cep_wav_1[tick+1];  
		}			
	}

	for (int frame=0; frame<FrmNum_wav_2; frame++) // ÿһ֡
	{		
		//�õ�һ֡����
		for (int j = 0; j < FrmLen; j++)
		{
			dBuff_wav_2[j] = (double)data_scaled_wav2[frame * FrmLen + j];
		}
		// ���һ֡�����ݽ��д���
		preemphasis(dBuff_wav_2, resultRaw, FrmLen);
		HammingWindow(resultRaw, dataRaw); 
		compute_fft(dataRaw, vecList);
		CFilt(dataRaw, FiltCoe1, FiltCoe2, Num, En,vecList);
		MFCC(En, Cep_wav_2);
		vecList.clear();

		for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
		{
			cep_part_2[frame*(PCEP-1)+tick] = Cep_wav_2[tick+1];
		}			
	}

	//printf("�� �����ƶ���֡��=%d\n", fram_move);
	for (int pole = 0; pole < fram_move; pole++)  
	{
		score[pole] = 0;
		for (int jj=0; jj<len_window*(PCEP-1); jj++)
		{
			float count = cep_part_1[jj] * cep_part_2[pole*(PCEP-1)+jj] / float(len_window*(PCEP-1));
			score[pole] += count;
					
		}		
		
	}

	float pos = findLocalMaximum(score, fram_move); // ���� len_window �� wav2�еĶ����λ�á�	
	bb = (double)(pos*FrmLen + (bias_wav_2 - bias_wav_1))/50.0/FrmLen;	

		
	delete [] data_wav1;
	delete [] data_wav2;      
	delete [] data_scaled_wav1;
	delete [] data_scaled_wav2 ;
	delete [] score;
	delete [] cep_part_1;
	delete [] cep_part_2;

	return 0;
}
#ifdef __cplusplus  
}  
#endif  

 

/*
	һ����������ĳ��ʱ����Ѱ��С���� 
*/
#ifdef __cplusplus  
extern "C" {  
#endif  
__declspec(dllexport)  int speechMatch(const char *file1 ,const char *file2, const double st_time, const double end_time, double &bb)
{  

	int ret = 0;
	FILE* fp_debug;		
	fp_debug = fopen("debug.txt","a+");
    
	//��ȡ��������ĳ��ȺͲ����� Ȼ��short����
	printf("read wav_1:%s\nread wav_2:%s\n", file1, file2);
	int len_wav_1 = ReadFileLength(file1, &sampleRate);
	int len_wav_2 = ReadFileLength(file2, &sampleRate);
	if(len_wav_1<=0 || len_wav_2 <= 0)
	{
		printf("read_file wav err!\n");
		return -1;
	}

	double time_len_wav_2 = end_time-st_time;	
	int bias = int(st_time*SAMPRATE);
	len_wav_2 = int(time_len_wav_2*SAMPRATE);
	printf("len_wav_1=%d[sizeof(short)]\n",len_wav_1);
	printf("bias=%d\tlen_wav_2=%d\n",bias,len_wav_2);

	//wav1��wav2����ҪС	
	if (sampleRate != SAMPRATE || time_len_wav_2 > double(len_wav_2)/16000.0 || time_len_wav_2 < double(len_wav_1)/16000.0 ) 
	{
		printf("[sampleRate:%d != 16000] or [len_wav_2:%d < len_wav_1:%d]!\n", sampleRate, len_wav_2, len_wav_1);	
		return -1;
	}
	


	short *data_wav1 = new short[len_wav_1];
	short *data_wav2 = new short[len_wav_2]; 
	ret = ReadFile(file1, data_wav1, 0, len_wav_1);	
	if(ret < 0)
	{
		printf("ReadFile:%s error!\n",file1);
		return 0;
	}

	ret = ReadFile(file2, data_wav2, bias, len_wav_2);
	if(ret < 0)
	{
		printf("ReadFile:%s error!\n",file2);
		return 0;
	}
	

	// һ֡=20ms	1s=50֡	�� һ֡=16*20��short
	int FrmNum_wav_1 = len_wav_1/FrmLen;
	int FrmNum_wav_2 = len_wav_2/FrmLen;
	printf("FrmNum_wav_1=%d\tFrmNum_wav_2=%d\n", FrmNum_wav_1, FrmNum_wav_2);
	printf("time_wav_1=%fs\ttime_wav_2=%fs\n", float(len_wav_1)/16000.0, float(len_wav_2)/16000.0);


	int len_window = FrmNum_wav_1;  // ��ȡ�� �Ա�֡ ����
	int fram_move = FrmNum_wav_2 - FrmNum_wav_1; // �����ƶ��ķ�Χ
	float *score = new float[fram_move];
	double *cep_part_1 = new double[(PCEP-1)*FrmNum_wav_1];
	double *cep_part_2 = new double[(PCEP-1)*FrmNum_wav_2];	

	float *data_scaled_wav1 = new float[len_wav_1];
	float *data_scaled_wav2 = new float[len_wav_2];
	
	DataScaling(data_wav1, data_scaled_wav1, len_wav_1);
	DataScaling(data_wav2, data_scaled_wav2, len_wav_2);
		
	
	InitHamming();//��ʼ��������
	InitFilt(FiltCoe1, FiltCoe2, Num); //��ʼ��MEL�˲�ϵ��
	double En[FiltNum+1];         //Ƶ������
	double Cep_wav_1[PCEP];//MFCC���
	double Cep_wav_2[PCEP];//MFCC���	
	
	for (int frame=0; frame<FrmNum_wav_1; frame++) // ÿһ֡
	{		
		//�õ�һ֡����
		for (int j = 0; j < FrmLen; j++)
		{
			dBuff_wav_1[j] = (double)data_scaled_wav1[frame*FrmLen+j];

		}
		// ���һ֡�����ݽ��д���
		preemphasis(dBuff_wav_1, resultReference, FrmLen); //Ԥ���ؽ������result����
		HammingWindow(resultReference, dataReference); //��һ֡���ݼӴ�,����data����
		compute_fft(dataReference, vecList);
		CFilt(dataReference, FiltCoe1, FiltCoe2, Num, En,vecList);
		MFCC(En, Cep_wav_1);
		vecList.clear();

		for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
		{
			// ÿһ֡������������
			cep_part_1[frame*(PCEP-1)+tick] = Cep_wav_1[tick+1];  
		}			
	}

	for (int frame=0; frame<FrmNum_wav_2; frame++) // ÿһ֡
	{		
		//�õ�һ֡����
		for (int j = 0; j < FrmLen; j++)
		{
			dBuff_wav_2[j] = (double)data_scaled_wav2[frame * FrmLen + j];
		}
		// ���һ֡�����ݽ��д���
		preemphasis(dBuff_wav_2, resultRaw, FrmLen);
		HammingWindow(resultRaw, dataRaw); 
		compute_fft(dataRaw, vecList);
		CFilt(dataRaw, FiltCoe1, FiltCoe2, Num, En,vecList);
		MFCC(En, Cep_wav_2);
		vecList.clear();

		for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
		{
			cep_part_2[frame*(PCEP-1)+tick] = Cep_wav_2[tick+1];
		}			
	}

	printf("�� �����ƶ���֡��=%d\n", fram_move);
	for (int pole = 0; pole < fram_move; pole++)  
	{
		score[pole] = 0;
		for (int jj=0; jj<len_window*(PCEP-1); jj++)
		{
			float count = cep_part_1[jj] * cep_part_2[pole*(PCEP-1)+jj] / float(len_window*(PCEP-1));
			score[pole] += count;
					
		}		
		//fprintf(fp_debug, "score[%d]=%f\n", pole, score[pole]);
	}

	// �Լ����Լ�ƥ�����õ����ֵ  
	float score_self = 0.0;
	for (int jj=0; jj<len_window*(PCEP-1); jj++)
	{
		float count = cep_part_1[jj] * cep_part_1[jj] / float(len_window*(PCEP-1));
		score_self += count;
					
	}		

	float pos = findLocalMaximum(score, fram_move);	
	int eee = int(1000*(score[int(pos)]/score_self));
	printf("\nscore_max=%.4f\tscore_self=%.4f\teee=%d\n", score[int(pos)], score_self, eee);

	bb = (double)(pos)/50.0;	
		
	delete [] data_wav1;
	delete [] data_wav2;      
	delete [] data_scaled_wav1;
	delete [] data_scaled_wav2 ;
	delete [] score;
	delete [] cep_part_1;
	delete [] cep_part_2;

	fclose(fp_debug);
	fp_debug = NULL;

	// ����ƥ��ȣ�
	return eee;
}
#ifdef __cplusplus  
}  
#endif  



/*
   ����������  ���� ���Ŷȵ� 
*/
#ifdef __cplusplus  
extern "C" {  
#endif  
__declspec(dllexport)  int speechMatch_conf(const char *file1 ,const char *file2, double &aa,double &bb)
{  

	int ret = 0;
	FILE* fp_debug;		
	fp_debug = fopen("debug.txt","a+");
    
	//��ȡ��������ĳ��ȺͲ����� Ȼ��short����
	printf("read wav_1:%s\nread wav_2:%s\n", file1, file2);
	int len_wav_2 = ReadFileLength(file2, &sampleRate);
	int len_wav_1 = ReadFileLength(file1, &sampleRate);
		
	//�������ݵĳ��Ȳ��ܴ���300s	wav1��wav2����ҪС	
	if (sampleRate != SAMPRATE || len_wav_1 > sampleRate*300 || len_wav_2 < len_wav_1 ) 
	{
		printf("[sampleRate:%d != 16000] or [len_wav_1:%d s > 300s] or [len_wav_2:%d < len_wav_1:%d]!\n", sampleRate, len_wav_1/16000, len_wav_2, len_wav_1);	
		return -1;
	}
	
	// ��bias����ʼ��ȡ len_wav_1 �� data_wav1 ��
	short *data_wav1 = new short[len_wav_1];
	short *data_wav2 = new short[len_wav_2]; 
	ret = ReadFile(file1, data_wav1, 0, len_wav_1);	
	if(ret < 0)
	{
		printf("ReadFile:%s error!\n",file1);
		return 0;
	}

	ret = ReadFile(file2, data_wav2, 0, len_wav_2);
	if(ret < 0)
	{
		printf("ReadFile:%s error!\n",file2);
		return 0;
	}
	printf("len_wav_1=%d\tlen_wav_2=%d [sizeof(short)]\n",len_wav_1, len_wav_2);

	// һ֡=20ms	1s=50֡	�� һ֡=16*20��short
	int FrmNum_wav_1 = len_wav_1/FrmLen;
	int FrmNum_wav_2 = len_wav_2/FrmLen;
	printf("FrmNum_wav_1=%d\tFrmNum_wav_2=%d\n", FrmNum_wav_1, FrmNum_wav_2);
	printf("time_wav_1=%fs\ttime_wav_2=%fs\n", float(FrmNum_wav_1)/50.0, float(FrmNum_wav_2)/50.0);


	int len_window = FrmNum_wav_1;  // ��ȡ�� �Ա�֡ ����
	int fram_move = FrmNum_wav_2 - FrmNum_wav_1; // �����ƶ��ķ�Χ
	float *score = new float[fram_move];
	double *cep_part_1 = new double[(PCEP-1)*FrmNum_wav_1];
	double *cep_part_2 = new double[(PCEP-1)*FrmNum_wav_2];	

	float *data_scaled_wav1 = new float[len_wav_1];
	float *data_scaled_wav2 = new float[len_wav_2];
	memset(data_scaled_wav1, 0 , sizeof(short)*len_wav_1);
	DataScaling(data_wav1, data_scaled_wav1, len_wav_1);
	DataScaling(data_wav2, data_scaled_wav2, len_wav_2);
		
	
	InitHamming();//��ʼ��������
	InitFilt(FiltCoe1, FiltCoe2, Num); //��ʼ��MEL�˲�ϵ��
	double En[FiltNum+1];         //Ƶ������
	double Cep_wav_1[PCEP];//MFCC���
	double Cep_wav_2[PCEP];//MFCC���	
	
	for (int frame=0; frame<FrmNum_wav_1; frame++) // ÿһ֡
	{		
		//�õ�һ֡����
		for (int j = 0; j < FrmLen; j++)
		{
			dBuff_wav_1[j] = (double)data_scaled_wav1[frame*FrmLen+j];

		}
		// ���һ֡�����ݽ��д���
		preemphasis(dBuff_wav_1, resultReference, FrmLen); //Ԥ���ؽ������result����
		HammingWindow(resultReference, dataReference); //��һ֡���ݼӴ�,����data����
		compute_fft(dataReference, vecList);
		CFilt(dataReference, FiltCoe1, FiltCoe2, Num, En,vecList);
		MFCC(En, Cep_wav_1);
		vecList.clear();

		for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
		{
			// ÿһ֡������������
			cep_part_1[frame*(PCEP-1)+tick] = Cep_wav_1[tick+1];  
		}			
	}

	for (int frame=0; frame<FrmNum_wav_2; frame++) // ÿһ֡
	{		
		//�õ�һ֡����
		for (int j = 0; j < FrmLen; j++)
		{
			dBuff_wav_2[j] = (double)data_scaled_wav2[frame * FrmLen + j];
		}
		// ���һ֡�����ݽ��д���
		preemphasis(dBuff_wav_2, resultRaw, FrmLen);
		HammingWindow(resultRaw, dataRaw); 
		compute_fft(dataRaw, vecList);
		CFilt(dataRaw, FiltCoe1, FiltCoe2, Num, En,vecList);
		MFCC(En, Cep_wav_2);
		vecList.clear();

		for (int tick = 0; tick<PCEP-1; tick++)  // ��һ֡��12��mfccϵ��
		{
			cep_part_2[frame*(PCEP-1)+tick] = Cep_wav_2[tick+1];
		}			
	}

	printf("�� �����ƶ���֡��=%d\n", fram_move);
	for (int pole = 0; pole < fram_move; pole++)  
	{
		score[pole] = 0;
		for (int jj=0; jj<len_window*(PCEP-1); jj++)
		{
			float count = cep_part_1[jj] * cep_part_2[pole*(PCEP-1)+jj] / float(len_window*(PCEP-1));
			score[pole] += count;
					
		}		
		//fprintf(fp_debug, "score[%d]=%f\n", pole, score[pole]);
	}

	float pos = findLocalMaximum(score, fram_move);	
	aa=1;
	bb = (double)(pos)/50.0;	

		
	delete [] data_wav1;
	delete [] data_wav2;      
	delete [] data_scaled_wav1;
	delete [] data_scaled_wav2 ;
	delete [] score;
	delete [] cep_part_1;
	delete [] cep_part_2;

	fclose(fp_debug);
	fp_debug = NULL;
	return 0;
}
#ifdef __cplusplus  
}  
#endif  


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 ��˹����  
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

/*
������Сֵ  
*/
float findLocalMaximum(float* score,int length)
{
	float maxiScore = 0;
	float maxiPos = 0;
	for (int i=0;i<length;i++)
	{
		if (score[i]>maxiScore)
		{
			maxiScore = score[i];		
			maxiPos = i;
			
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
		//MFCCcoefficient.push_back(Cep[idcep]);
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

/*
FFT 
*/
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

/*
����FFT
*/
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

void DataScaling(short* data, float* dataScaled, int halfWindow)
{

	float maxValue = 0;
	//for (int i=0;i<halfWindow*2-1;i++)
	for (int i=0;i<halfWindow-1;i++)
	{
		if (data[i]<maxValue)
		{
			maxValue = data[i];
		}
	}
	
	//printf("maxValue=%f\n",maxValue);
	
	for (int i=0;i<halfWindow;i++)
	{
		dataScaled[i] = data[i]/maxValue;
	}
}

