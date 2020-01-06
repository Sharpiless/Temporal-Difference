使用Detertor类的catch_video实现，具体参数：

1. video_index：摄像头索引（数字）或者视频路径（字符路径）
2. k_size：中值滤波的滤波器大小
3. iteration：腐蚀+膨胀的次数，0表示不进行腐蚀和膨胀操作
4. threshold：二值化阙值
5. bias_num：计算帧差图时的帧数差
6. min_area：目标的最小面积
7. show_test：是否显示二值化图片


效果如图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020010615542185.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

