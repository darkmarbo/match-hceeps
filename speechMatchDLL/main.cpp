#include"speechMatch.h"

void main(int argc, char* argv[])
{
	FILE* ofp;
	double aa = 0.0;
	double bb = 0.0;	
	
	if(argc!=4)
	{
		printf("Please input arguments: wav1 wav2 out_file\n");
		printf("//wav1: 标杆数据，即其他数据要向这个数据进行对齐\nwav2:带对齐数据，该程序完成的功能是计算该数据与标杆数据的时间差\nwav2:结果文件，其中保存在几个关键时间点上待对齐数据与标杆数据的时间差");
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