int speechMatch_a(const char *file1, const char *file2, double &aa, double &bb)
	参数：
		file1和file2为输入两个语音的文件名。
		aa和bb是计算得到的斜率和延迟。
		
	功能：
		针对不同设备的时钟频率不一致，使用线性规划求y=ax+b，a和b为函数求得的值。
		解释：file1的x时间点，对应file2的ax+b时间点。
	返回值：
		0：正常，
		-1：采样率
		-2：打开文件错误
		-3：文件头错误
		-4：采样点数不一致
		-5：读取的采样点超出范围
		-6：其他读取文件错误
		1:匹配结果不准确（延迟超过5分钟、两个文件长度相差20分钟等……）