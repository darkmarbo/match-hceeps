#include"speechMatch.h"

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