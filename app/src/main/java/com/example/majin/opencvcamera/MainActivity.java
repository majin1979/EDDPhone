package com.example.majin.opencvcamera;
import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.provider.ContactsContract;
import android.util.Log;
import android.widget.TextView;
import java.lang.Math;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.LineSegmentDetector;

import static android.util.Log.VERBOSE;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.LSD_REFINE_ADV;
import static org.opencv.imgproc.Imgproc.clipLine;
import static org.opencv.imgproc.Imgproc.line;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private String TAG = "ElevatorDetectorModel";
    //OpenCV的相机接口
    private CameraBridgeViewBase mCVCamera;
    //缓存相机每帧输入的数据
    private Mat mRgba,mGray,mLineL,mLineR,mPointL,mPointR;
    //private  MatOfPoint mPointL,mPointR;
    private Mat mCanny,mLine,mROI,mclose,mopen;
    //private LineSegmentDetector LSD;
    private int swid,shei;
    private Rect mrect;
    private double[] areasize = new double[3];
    //private double[] closecounter = new double[10];
    private byte lines[][][][];
    private int Var[][];
    private int length[][];
    //private int Speed[][][];
    private int framecc,cacucc,linenum,framenum,thresvalue,widthvalue;
    private int linexyc,PLRmin,LRreducemin,LRreducemax;
    private int closecc,closehold,outtimeframe;
    private double closelast,middlevalue,backvalue;
    private boolean fillfull,varpoint,haveline,linecacu,MiddleIdle;
    private long startTime,consumingTime;
    private TextView tvL,tvR;
    private String DisStrL,DisStrR,logtag;
    Point PLT,PLB,PRT,PRB;
    //创建一个Handle对象
    final Handler handler = new Handler();

    //创建一个Runnable对象，在对象中更新UI文本
    final Runnable runnableL  = new Runnable() {
        public void run() {
            tvL.setText(DisStrL);
        }
    };
    final Runnable runnableR  = new Runnable() {
        public void run() {
            tvR.setText(DisStrR);
        }
    };

    /**
     * 通过OpenCV管理Android服务，异步初始化OpenCV
     */
    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    mCVCamera.enableView();
                    break;
                default:
                    break;
            }
        }
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //初始化输出文本
        DisStrL = DisStrR = "设备环境初始化中";
        tvL = findViewById(R.id.textViewL);
        tvL.setText(DisStrL);
        tvR = findViewById(R.id.textViewR);
        tvR.setText(DisStrR);
        handler.post(runnableL);
        handler.post(runnableR);

        //赋值摄像头对象
        mCVCamera = findViewById(R.id.camera_view);
        mCVCamera.setCvCameraViewListener(this);
        // 打开USB摄像头 ID=0
        //mCVCamera.setCameraIndex(-1);
    }
    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV library not found!");
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

    }

    @Override
    public void onDestroy() {
        if(mCVCamera!=null){
            mCVCamera.disableView();
        }
        super.onDestroy();
    }

    private void deviation(byte datas[][][][],int vars[][]){
        //linev 代表不同的采样行号
        for(int linev = 0;linev<linenum;linev++) {
            //linevc 代表最近的framenum帧的标号
            for(int pixrows = 0;pixrows<swid/2;pixrows++){
                int vartotal = 0;//单点数据framenum帧的总和
                for(int linevc = 0;linevc<framenum;linevc++){
                    vartotal += datas[linevc][linev][pixrows][0];//计算单点数据framenum帧的总和
                }
                int varaver = vartotal/framenum;//计算单点数据framenum帧的平均值
                int varsigma = 0;
                for(int linevc = 0;linevc<framenum;linevc++){
                    //计算单点与平均值差的平方的framenum帧总和
                    varsigma += (datas[linevc][linev][pixrows][0]-varaver)*(datas[linevc][linev][pixrows][0]-varaver);
                }
                vars[linev][pixrows] = (int)Math.sqrt(varsigma/framenum); //计算需要的方差
            }
        }
    }
    private void var2length(int vars[][],int lengths[][],Mat PL,Mat PR,int threshold,int width){
        //此函数中，注释了MatofPoint方式的数据填充方式，改用手动put数据。
        //List pointsL = new ArrayList<Point>();
        //List pointsR = new ArrayList<Point>();
        for(int rows = 0;rows<linenum;rows++){
            int leftlength = 0;
            int rightlength = 0;
            //Point temppoint;
            int temppoint[] = new int[2];
            boolean leftrun,rightrun;
            leftrun = rightrun= true;
            //区域验证：假设宽度swid=640 则swid/2 = 320 swid/4 = 160 width = 5 pixs=5~160 wp=0～4
            //则：左侧var第二下标 第一次计算为155～159 最后一次计算为0～4

            //则：右侧var第二下标 第一次计算为164～160 最后一次计算为319～315

            for(int pixs = width;pixs<=swid/4;pixs++){
                int ltotal = 0;
                int rtotal = 0;
                for(int wp = 0;wp<width;wp++){
                    if(leftrun) ltotal += vars[rows][swid/4-pixs+wp];
                    if(rightrun) rtotal += vars[rows][swid/4+pixs-wp-1];
                }
                if(leftrun) leftlength = pixs-width;//当前计算区域右侧边界距离中缝的长度即为右距=左距（pixs）减去宽度
                if(ltotal>threshold*width) leftrun = false;//如果width宽度中，平均值大于阀值，则左侧探测停止。

                if(rightrun) rightlength = pixs-width-1;//当前计算区域左侧边界距离中缝的长度即为左距=右距（pixs）减去宽度再减1，因为右侧起始点为swid/2+1
                if(rtotal>threshold*width) rightrun = false;//如果width宽度中，平均值大于阀值，则右侧探测停止
                if(!(leftrun||rightrun)) break;//如果左右均探测到波动边界，则跳出循环。

            }
            //speeds[rows][0] = leftlength - lengths[rows][0];//将当前左长度减去上一次计算的长度，即为当前左速度（左长度差/帧）
            lengths[rows][0] = leftlength;//将当前左长度存入左长度缓存
            temppoint[0] = swid/2-leftlength-1;
            temppoint[1] = rows*shei/(linenum*3);
            mPointL.put(rows,0,temppoint);
            //temppoint = new Point(swid/2-leftlength-1,rows*shei/(linenum*3));
            //pointsL.add(temppoint);

            //speeds[rows][1] = rightlength - lengths[rows][1];//将当前右长度减去上一次计算的长度，即为当前右速度（右长度差/帧）
            lengths[rows][1] = rightlength;//将当前右长度存入长度缓存
            temppoint[0] = swid/2+rightlength;
            temppoint[1] = rows*shei/(linenum*3);
            mPointR.put(rows,0,temppoint);
            //temppoint = new Point(swid/2+rightlength,rows*shei/(linenum*3));
            //pointsR.add(temppoint);
        }
        //mPointL.fromList(pointsL);
        //mPointR.fromList(pointsR);
    }
    @Override
    public void onCameraViewStarted(int width, int height) {
        //统一在此定义初始化参数
        logtag = "EDDPhone";
        if(width==0||height==0) {
            DisStrL = "UVC摄像头，固定分辨率640*480";
            width = 640;//USB摄像头需要自定义分辨率（画面宽高）
            height = 480;
        }
        else DisStrL = String.format("原生摄像头，分辨率%d*%d",width,height);
        Log.w(logtag,DisStrL);

        swid = width;//原生摄像头可以自动获取分辨率（画面宽高）
        shei = height;
        linenum = 10;//采样线的数量，居中水平横线，宽度为画面一半，纵坐标从top开始，至上方1/3处，均分。平衡性能和效率的参数。
        framenum = 10;//采样的帧数，同时会根据此值定义缓存数量。平衡性能和效率的参数。尽量为偶数
        thresvalue = 15;//方差的阀值，超过此值，则判定为实际变化。影响判定的重要参数。
        widthvalue = 20;//阀值的计算宽度，宽度的像素方差均值大于阀值，则判定为变化。影响判定的重要参数。

        PLRmin = 100;//第一个左右边符合斜率的梯形的上下底和（因为高都相等，底和也就等同面积---比值相等）
        LRreducemin = 20;//第二、第三帧的梯形底和与前一帧的差需要控制在阀值内，此为最小阀值。
        LRreducemax = 200;//第二、第三帧的梯形底和与前一帧的差需要控制在阀值内，此为最大阀值。

        outtimeframe = 100;//中缝对比时，当总帧数大于此阀值，则检测停止，流程返回到之前的关门初期探测。
        middlevalue = 0.2;//中缝对比检测时，当对比值小于此阀值，则进入最终关门状态的连续三帧检测，如果大于，则进行递减检测。
        backvalue = 0.1;//中缝对比检测进行递减检测时，考虑到误差和干扰，偶尔会有极小值的逆向递增现象出现，采用此阀值限定递增不能超过的区间。

        //参数定义结束
        //LSD = Imgproc.createLineSegmentDetector();
        mRgba = new Mat(height, width, CvType.CV_8UC4); //原始RGBA四通道图像（携带Alpha透明度信息的PNG图像）
        mGray = new Mat(height, width, CvType.CV_8UC4); //灰度图像
        mPointL = new Mat(linenum,1,CvType.CV_32SC2);   //左点阵合成的Mat矩阵，以便FitLine函数拟合
        mPointR = new Mat(linenum,1,CvType.CV_32SC2);   //右点阵合成的Mat矩阵，以便FitLine函数拟合
        //经测试，MatofPoint方式填充的点位图与手动put方式填充的点位图输入FitLine后效果一样。
        //mPointL = new MatOfPoint();
        //mPointR = new MatOfPoint();
        mLineL = new Mat();   //左点阵合成的Mat矩阵，以便FitLine函数拟合
        mLineR = new Mat();   //右点阵合成的Mat矩阵，以便FitLine函数拟合

        framecc = cacucc = 0;//初始化帧数计数器和同步计数器
        linexyc = 0;
        closelast = 0;
        fillfull = varpoint = haveline = linecacu = false;//初始化填充帧完成标志
        MiddleIdle = true;
        lines = new byte[framenum][linenum][swid/2][1];//framenum帧缓存 linenum行采样线 画面宽度一半的采样宽度 1个像素*1个字节的采样数据
        Var = new int[linenum][swid/2];//linenum行采样线的帧之间方差数据，画面一半宽度的采样像素
        length= new int[linenum][2]; //当前linenum行采样线的中间相对静止区域的长度,两个方向
        //linenum行采样线，两个方向的静止区域变化速度，具备正负值，正表示增，负表示减(20190107取消速度计算)
        //Speed = new int[framenum][linenum][2];
        //startTime = System.currentTimeMillis();//计算粗略的毫秒时间
        Thread threadvar = new Thread("ThreadVar") {
            public void run() {
                while(true) {
                    try {
                        if (fillfull&&MiddleIdle) {
                            int sleeptime = 0;
                            while(cacucc==framecc){//等待新的帧存入缓冲区
                                Thread.sleep(1);
                                sleeptime++;
                            }
                            startTime = System.currentTimeMillis();//计算粗略的毫秒时间

                            deviation(lines, Var);//将lines中的像素数据计算出方差存入Var中。
                            /**将Var中的方差数据,根据上一帧长度length数据计算出中间相对阀值threshold的
                             //静止区域(超越宽度width)的增减速度，并将速度数据存入Speed的当前(十帧)的速度数据中。
                             //完成计算后，将length更新为新的数据。**/
                            //20190107 更改为计算中间相对阀值静止的长度数据,并将左右极限点分别写入左右以Point命名的Mat中
                            var2length(Var,length,mPointL,mPointR,thresvalue,widthvalue);

                            cacucc = framecc;//更新同步标志
                            if(cacucc == framenum-1){
                                varpoint = true;//速度数据初始化完毕
                            }
                            if(varpoint){
                                Imgproc.fitLine(mPointL,mLineL,Imgproc.CV_DIST_L2,0, 0.01, 0.01);
                                Imgproc.fitLine(mPointR,mLineR,Imgproc.CV_DIST_L2,0, 0.01, 0.01);
                                linecacu = true;
                                consumingTime = System.currentTimeMillis() - startTime;//计算粗略的毫秒时间
                                DisStrR = String.format("等待帧填充时间%dms 帧处理时间%dms",sleeptime,(int)consumingTime);
                            }
                            else{
                                DisStrR = String.format("每帧处理耗时%dms",sleeptime);
                            }
                        } else {
                            Thread.sleep(100);
                            DisStrR = "数据读取中";
                        }
                        handler.post(runnableR);
                    }
                    catch (InterruptedException e) {
                        // TODO 自动生成的 catch 块
                        e.printStackTrace();
                    }
                }
            }
        };
        threadvar.start();
    }

    @Override
    public void onCameraViewStopped() {
        // TODO Auto-generated method stub
        //mRgba.release();
        //mGray.release();
    }

    /**
     * 图像处理都写在此处，未使用代码注释在ScreenOffAdminReceiver中
     */
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Imgproc.cvtColor(mRgba, mGray, COLOR_BGR2GRAY);
        if(MiddleIdle) {
            int getn;
            for (int rows = 0; rows < linenum; rows++) {
                for (int i = 0; i < swid / 2; i++) {
                    getn = mGray.get(rows * shei / (linenum * 3), i + swid / 4, lines[framecc][rows][i]);
                    if (getn == 0) {
                        DisStrL = "摄像头读取错误";
                        break;
                    }
                }
            }
            //设置framenum帧缓冲，下标framecc到framenum-1手动还原到0，并且在算法线程中将framecc作为同步指示器，检测到framecc更新则开始新一轮的计算
            if (framecc >= (framenum - 1)) {
                framecc = 0;
                fillfull = true;//填充帧完成
            } else framecc++;

            if (linecacu) {
                int N = mLineL.checkVector(1);
                double linesLR[][][] = new double[2][4][1];
                // Draw segments
                for (int i = 0; i < N; ++i) {
                    linesLR[0][i] = mLineL.get(i, 0);
                    linesLR[1][i] = mLineR.get(i, 0);
                }
                if ((-1 < linesLR[0][0][0] / linesLR[0][1][0]) && (linesLR[0][0][0] / linesLR[0][1][0] < 1) && (1 > linesLR[1][0][0] / linesLR[1][1][0]) && (linesLR[1][0][0] / linesLR[1][1][0] > -1)) {
                    //由FitLine计算出来的二维Line具备4个float值，Vx Vy x0 y0，分别为 两个维度的直线向量与直线上某点坐标。
                    //其满足等式  (X-x0)/Vx = (Y-y0)/Vy
                    //当确定Y值时 X = x0 + (Y-y0)*Vx/Vy 特别当Y=0时 X = x0 - y0*Vx/Vy
                    PLT = new Point(linesLR[0][2][0] + (2 - linesLR[0][3][0]) * linesLR[0][0][0] / linesLR[0][1][0], 2);
                    PLB = new Point(linesLR[0][2][0] + ((linenum - 1) * shei / (linenum * 3) - linesLR[0][3][0]) * linesLR[0][0][0] / linesLR[0][1][0], (linenum - 1) * shei / (linenum * 3));
                    PRT = new Point(linesLR[1][2][0] + (2 - linesLR[1][3][0]) * linesLR[1][0][0] / linesLR[1][1][0], 2);
                    PRB = new Point(linesLR[1][2][0] + ((linenum - 1) * shei / (linenum * 3) - linesLR[1][3][0]) * linesLR[1][0][0] / linesLR[1][1][0], (linenum - 1) * shei / (linenum * 3));
                    haveline = true;
                    linexyc++;//符合斜率计数器累加
                    DisStrL = "进入底和状态第一帧";
                } else {
                    linexyc = 0;//符合斜率计数器清零
                }
                if (haveline) {
                    line(mRgba, PLT, PLB, new Scalar(255, 0, 0), 2);
                    line(mRgba, PRT, PRB, new Scalar(0, 255, 0), 2);
                    line(mRgba, PLT, PRT, new Scalar(0, 0, 255), 2);
                    line(mRgba, PLB, PRB, new Scalar(0, 0, 255), 2);
                    if (linexyc > 0) {
                        areasize[linexyc - 1] = PRT.x - PLT.x + PRB.x - PLB.x;//将当前底和填入area0
                        if (linexyc == 1) {
                            if (areasize[0] < PLRmin) linexyc = 0;//如果底和小于阀值，则归零
                        }
                        //如果第二与第一底和的差不在阀值范围，则将第二底和回退到第一底和，如果第二底和也不符合阀值，则清零
                        if (linexyc == 2) {
                            if (((areasize[0] - areasize[1]) < LRreducemin) || ((areasize[0] - areasize[1]) > LRreducemax)) {
                                if (areasize[1] > PLRmin) {
                                    DisStrL = "从底和状态第二帧回退到第一帧";
                                    areasize[0] = areasize[1];
                                    linexyc = 1;
                                } else {
                                    DisStrL = "从底和状态第二帧退出";
                                    linexyc = 0;
                                }
                            }
                        }
                        //如果第三与第二底和的差不在阀值范围，则将第三底和回退到第一底和，如果第三底和也不符合阀值，则清零
                        if (linexyc == 3) {
                            if (((areasize[1] - areasize[2]) < LRreducemin) || ((areasize[1] - areasize[2]) > LRreducemax)) {
                                if (areasize[2] > PLRmin) {
                                    DisStrL = "从底和状态第三帧回退到第一帧";
                                    areasize[0] = areasize[2];
                                    linexyc = 1;
                                } else {
                                    DisStrL = "从底和状态第三帧退出";
                                    linexyc = 0;
                                }
                            }
                            //如果三底和都符合，则判断关门正在进行，停止动态方差分析，进入门中缝区域对比流程
                            else {
                                //清零底和差计数器，完成底和差阀值对比工作，切换flag值，进入中缝对比循环。
                                linexyc = 0;
                                MiddleIdle = false;
                                closecc = 0;
                                closehold = 0;
                                mrect = new Rect((int)PLT.x,0,(int)(PRT.x-PLT.x),shei/3);
                                mclose = mGray.submat(mrect);
                                mclose.convertTo(mclose, CvType.CV_32F);
                                DisStrL = "检测到门正在关闭";
                            }
                        }
                    }
                }
            }
        }
        //进入门中缝区域对比流程
        else{
            Imgproc.rectangle(mRgba,PLT,new Point(PRT.x,shei/3),new Scalar(255,255,0),3);
            mROI = mGray.submat(mrect);
            mROI.convertTo(mROI, CvType.CV_32F);
            double closecom = Imgproc.compareHist(mROI, mclose, Imgproc.CV_COMP_CORREL);
            if(closecc>0) {
                //从下一帧开始，到接近关门完成之前，如果出现逆差或者长时间不关门完成，则退出中缝对比。
                if(closecom > middlevalue){//大区间
                    if(closecom-closelast>backvalue||closecc>outtimeframe) {//逆或超,其中涵盖了进入小区间累加不足3次又跳回大区间的情况=逆
                        DisStrL = "逆了或者超了";
                        MiddleIdle = true;
                    }
                    //不逆不超，继续下一帧
                }
                else{//小区间
                    if(closecc<3){//太短也不正常？
                        DisStrL = "太短帧数";
                        MiddleIdle = true;
                    }
                    else{//等三次累加
                        DisStrL = "关门累加";
                        closehold++;
                    }
                    if(closehold>3){
                        MiddleIdle = true;
                        DisStrL = "门已经关闭";
                    }
                }
                DisStrR = String.format("对比度:%g 中缝判断总帧数:%d 关门判断帧数:%d",closecom,closecc,closehold);
            }
            else{
                DisStrL = "进入中缝对比";
            }
            closelast = closecom;
            closecc++;
        }
        Log.w(logtag,DisStrL);
        Log.i(logtag,DisStrR);
        handler.post(runnableL);
        handler.post(runnableR);
        return mRgba;//此处必须返回与控件相同分辨率的mat，否则无法显示。
    }
}