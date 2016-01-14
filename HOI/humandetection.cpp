#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2\opencv.hpp"
#include <time.h> 
#include <stdio.h>
#include <windows.h> 
#include <opencv2\ml\ml.hpp>
#include "svm.h"
#include <list>
#include <labeling.h>
#include "C:\Users\mitsuhashi\Desktop\lab\program\CvHMM\CvHMM.h"
#include <fstream> 

#ifdef _DEBUG
    //Debugモードの場合
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_core230d.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_imgproc230d.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_highgui230d.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_objdetect230d.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_contrib230d.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_features2d230d.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_flann230d.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_gpu230d.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_haartraining_engined.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_legacy230d.lib")
  //  #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_ts230d.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_video230d.lib")
	#pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\libsvm.lib")
#else
    //Releaseモードの場合
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_core230.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_imgproc230.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_highgui230.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_objdetect230.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_contrib230.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_features2d230.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_flann230.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_gpu230.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_haartraining_engined.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_legacy230.lib")
 //   #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_ts230.lib")
    #pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\opencv_video230.lib")
	#pragma comment(lib,"C:\\OpenCV2.3\\build\\x86\\vc10\\lib\\libsvm.lib")
#endif
 
#define OPENCV_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
#define OPENCV_VERSION_CODE OPENCV_VERSION(CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION)
using namespace cv;
using namespace std;
//struct svm_model *model;

 HOGDescriptor hog;
 vector<Rect> found;

 int lengthseqs[10];//HMMで用いる10フレームでの距離を格納
 int svmnum=0;
 //入力するデータセット
 int dataend[17]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
 
 

struct svm_parameter param;		// SVM設定用パラメータ
struct svm_problem prob;		// データセット（ラベル＆データ）・パラメータ
struct svm_node *x_space;		// データ・パラメータ（svm_problemの下部変数）
struct svm_model *model;		// 学習データ・パラメータ

#define NUM_OF_DATA_SET	1186	// データセット数 
#define MAX_INDEX	15	// １データセットに収納されているデータ数（データ次元）
#define movienum 16
#define startnum 1 //動画のインデックス番号

int svmlabel=1;
int m=0;
int MAX_INDEX2=6; 
float svmthread=0;
//人物形状の切り取りの領域決定係数
float HOGx=0.1;
float HOGwidth=1;
float HOGy=0.1;
float HOGheight=0.1;

//腕の位置座標を読み込むファイル
char handcoorfile[150]="C:\\Users\\mitsuhashi\\Desktop\\lab\\program\\pose_estimation_code_release_v1.21\\example_data\\imgposecoordinates2.txt";
//頭部の位置座標を読み込むファイル
char headcoorfile[150]="C:\\Users\\mitsuhashi\\Desktop\\lab\\program\\pose_estimation_code_release_v1.21\\example_data\\headcoordinates20140718.txt";
char svmfile[50]="20140717.txt";


//l・・・各動画中のフレーム番号
//framenum・・・入力した全動画中のフレーム番号





 double  svmdetection(int *q){
 int a[5];


 svm_node z[16];
 int k;
 for(k=0;k<15;k++){
 z[k].index=k+1;
 z[k].value=*q;
 //printf("p=%d\n",*q);
 //printf("index=%d\n",z[k].index);
 //printf("value=%f\n",z[k].value);
 q++;
 }
 z[k+1].index=-1;
 printf("predict=%f\n",svm_predict(model,z));

 return svm_predict(model,z);
 
 }


 void calcdistance(){
 
 
 
 }





vector<Rect> humandetection(Mat img2){
	
  hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
  //矩形クラスかつ動的配列 found を定義
		  
		  //Rectはopencvのクラステンプレート
		  // 画像，検出結果，閾値（SVMのhyper-planeとの距離），
		  // 探索窓の移動距離（Block移動距離の倍数），
		  // 画像外にはみ出た対象を探すためのpadding，
		  // 探索窓のスケール変化係数，グルーピング係数


  hog.detectMultiScale(img2, found, svmthread, cv::Size(4,4), cv::Size(0,0), 1.5, 1);//検出窓可変で物体検出
		  //画像のデータ,検出する矩形領域,特徴点とSVM分割平面の閾値,検出窓の移動量,CPUインタフェースとの互換性を保つためのモックパラメータ,探索窓のスケール変化係数,重なりの区別を行う閾値
  
	
 
  return found;

}


 

void svm(vector< vector<float> > arr){

	
	int i,j=0;
    param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	//param.gamma = 1./MAX_INDEX2;
	param.gamma =0.0019;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 8192;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	FILE *fpp;

	//fpp=fopen("");

	//データのラベルを読み込み
	

	fpp=fopen(svmfile,"w");

	// データセットのパラメータ設定
	prob.l=dataend[movienum]+1;	// 入力する動画数、dataendの各要素には対応する動画の最後のフレームが格納、[movienum]には全動画の総フレーム
	prob.l=cv::saturate_cast<int>(arr.size());

	// 各パラメータのメモリ領域確保
	prob.y= new double[prob.l];		// ラベル
	prob.x= new svm_node *[prob.l];		// データセットの分だけデータ収納空間を作成
	x_space = new svm_node[(MAX_INDEX2+1)*prob.l];

	printf("start data insert\n");
	// データセット・パラメータへの数値入力
	for(i=0;i<prob.l;i++){
	//	printf("next dataset\n");
		//prob.y[i]=(int)data[i][0];	// ラベル入力
		if(i<NUM_OF_DATA_SET){
			prob.y[i]=1;
		//	printf("1");
		}
		else{
		prob.y[i]=0;
		//printf("0");
		}
		fprintf(fpp," %d ",(int)prob.y[i]);

		printf("i=%d\n",i);
		for(j=0;j<MAX_INDEX2;j++){

			//フレームナンバーにラベルを対応づけておく
			x_space[(MAX_INDEX2+1)*i+j].index = j+1;// データ番号の入力
			//x_space[(MAX_INDEX+1)*i+j].value = data[i][j+1];	// データ値の入力
			x_space[(MAX_INDEX2+1)*i+j].value = arr[i][j];

			fprintf(fpp," %d:%f ",j+1,arr[i][j]);
 		}
			fprintf(fpp,"\n");
 	
		x_space[(MAX_INDEX2+1)*i+MAX_INDEX2].index = -1;
		prob.x[i] = &x_space[(MAX_INDEX2+1)*i];	// prob.xとx_spaceとの関係付
	}						

	fclose(fpp);
	
	model = svm_train(&prob,&param);	// SVM学習
	svm_save_model("model.txt.model",model);	// 学習結果のファイル出力
	
	svm_node z[3];
	
	for(j=0;j<3;j++){
		
	for( i=0;i<7;i++){
		
		z[j].value=6;

	}

//	double d=svm_predict(model,z);
	}
	printf("学習終了→「model.txt」をファイル出力\n");
//	Sleep(100);

	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);

	// メモリ領域の開放
	delete[] prob.y;
	delete[] prob.x;
	delete[] x_space;
}






int
main(int argc, char *argv[]){
	
	FILE *fp,*fpp,*fp1; 
	vector< vector<float> > arr;
	int q;
	float surfrange=0.5;
	int l=0;
	char imagename[20];

	Mat img,imgprev,imgprev2,imgprev3;
	Mat prev1, next1,prev2,prev22,prev33;

	//opticalflowの値設定
   int level =3;
   double pyrScale=0.5;
   int winsize=20;
   int iterations=3; 
   int polyN=7;//(7,1.5)or(5,1.1)
   double polySigma=1.5;


  //Optical Flowのラベル数
   int clustnum=8;

	//学習するデータを読み込む
  //SURFのキーポイントの番号

   //フレーム格納
   int itrfr=0;
   int itrfrend=0;
   //各オプティカルフローのクラスタに属しているSURFキーポイントの数
   int labelcount[8];

   //SURFキーポイントの最終的なクラスタ
   int labels1[8];
   int max,maxlabel=0;	
   int i;
   int foundnum=0;
   int windownum=0;
  //オプティカルフローの番号カウント
 
//   cv::Mat p1flow(next.size(),CV_32F);
 // cv::Mat p2flow(next.size(),CV_32F);

  int rownum;
   int frame;
  int surfclusnum=15;
  int histall[15];
  float cossimilar[8];

  Size flowSize(50,50);

  arr.resize(1);
  arr[0].resize(6);

 // system("\"C:\\Program Files\\MATLAB\\R2013a\\bin\\matlab.exe -r chgdirectory.m\"");

  // chgdirectory.m

//  system("\"C:\\Users\\mitsuhashi\\Desktop\\lab\\program\\pose_estimation_code_release_v1.21\\code\\matlab.exe -r startup.m\"");

//動画C:\\Program Files\\MATLAB\\R2013a\\bin

  //データのチェック
//	 checkdata();


float t=0;
  float tim=0;

  
  fp=fopen("simcalcresult.txt","r");
  for(int k5=0;k5<22547;k5++){
  fscanf(fp," %f\n",&tim);
  t=t+tim;
  
  }
 
  t=t/22547;
  printf("t=%f",t);
  fclose(fp);

  fp=fopen("simcalcresult_not.txt","r");
  t=0;
   for(int k5=0;k5<24155;k5++){
  fscanf(fp," %f\n",&tim);
  t=t+tim;
  
  }
   t=t/24155;
   printf("t=%f",t);
   fclose(fp);
 // }

	int swt=0;
	scanf("%d",&swt);


   //読み込みファイルのダウンロード
   

   int framenum=0;
   Mat img1;
	vector<cv::Rect>::const_iterator it3;
	int n=0;
	float a[17][4];

	 fp=fopen(handcoorfile,"r");
	//if(swt==1){
	 for(n=0;n<movienum;n++){

	 VideoCapture cap;
	 char moviename[30];
	sprintf(moviename,"pp%d.avi",n+startnum);
	cap.open(moviename);
	printf("movie start::n=%d",n+startnum);
	if(n==16){
	
		FILE *fppp;
		
	fppp=fopen("negativestart.txt","w");
    fprintf(fppp,"%d",framenum);
	fclose(fppp);
	}
	//20140430_144444
	//20140501_160308
	//20140507_151427
	//20140507_155401
	//20140611_135438
	//20140611_145604
	//AVSS_AB_Hard_Divx.avi
	
	//開始フレーム調節
	//cap.set(CV_CAP_PROP_POS_FRAMES,300);

	
     //フレーム番号を初期化
	 l=0;
	   int a,b,c;

	 if(swt>1){
       

	 
	 

	 //  fp1=fopen("intractionframe.txt","r");
	//	fscanf(fp1,"%d %d %d",&c,&a,&b);

		cout<<"test"<<endl;


   }



while(1)
    {
	   //////////動画/////////////
	
		Mat befimage2;
		img1.copyTo(befimage2);
		//imgにcapを入力
        cap >> img1;
		//書き込み
		imwrite("image.jpg",img1);

		//img・・・最新フレーム
		//imgprev・・・ひとつ前のフレーム
		//img2prev・・・2つ前のフレーム

		Mat img=imread("image.jpg");

		framenum++;
	



	//	Mat mask=img-befimage2;
		//imshow("miru",img);
		//imshow("mask",mask);
		vector< vector<int> >  locationseq(10, vector<int>(1));
      //  char framedata[30];
		//////////動画/////////////*/




		if(l<3){
		imgprev=img;
		if(l>0){
		imgprev2=imgprev;
		}
		if(l>1){
		imgprev3=imgprev2;
		}

		l++;
		continue;
		}
	
		
		
		if(swt==1){
			if( img.empty() ){//imgが０なら停止の再帰判定
            printf("cant open image file\n");
			fpp=fopen("intractionframe.txt","a+");
	//			scanf("%d %d",&itrfr,&itrfrend);
			fprintf(fpp,"%d %d\n",n,framenum);
			fclose(fpp);
				break;
			}

	//	imshow("movie",img);
		cout<<"l="<<l<<endl;

		l++;
		

		continue;
		}
		
		

			if( img.empty() ){//imgが０なら停止の再帰判定
            printf("cant open image file\n");
			printf("%d",l);
				break;
			}
    			
	 //namedWindow("hazime", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	 //imshow("hazime", imgprev); 	
		
		//it3 = found.begin();

		//	Mat zenkei=img-imgprev;
		//	imwrite("zenkei.png",zenkei);



		//10フレームごとに人検出、found更新,初期フレームの更新
	if(l%10==0&&l>=10){

	//	if(l==78){
		found=humandetection(img);
		
		cout<<"found="<<found.size()<<endl;
		//人検出
		//  cout << "found:" << found.size() << endl;//検出数の出力

		foundnum=cv::saturate_cast<int>(found.size());
		// char matname[20];
		  printf("l=%d\n",l);


		  //人物位置情報シークエンスの末尾に0を挿入
		//  for(i=0;i<found.size();i++){
		//  locationseq[i].push_back(0);
		 // }


		  //区域の画像の情報を保持する
		 
		  //Sleep(1000);
		}

	//イテレータの初期化


	it3 = found.begin();

	//検出窓の表示番号の初期化
	windownum=0;
	int humannum=0;

	for(; it3!=found.end(); it3++) {
   
	
	cv::Rect r = *it3;
	
		cv::Mat label;
        cv::Mat center2;
		cv::Mat resizepic;
		vector<KeyPoint> kp_vec;//keypoint型の動的配列kp_vecを定義
        vector<float> desc_vec,prevsurf;   
        cv::Mat  centroids_;
		
		int k2,k3=0;
		int opcount=0;
		//cos類似度の最大値を格納
		float cosmax=0;
		//cos類似度を計算するために格納

		//cos類似度が最大になったラベルの番号を格納
		int coslabel=0;
			// 描画に際して，検出矩形を若干小さくする			
			r.x+= cvRound(r.width*0.1);
			//r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width);
		
		    r.y+= cvRound(r.height*0.07);
			//r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			

			////////////////////////////任意の画像サイズに切りとる場合////////////////////////////////////
			/*r.width = 320;
			r.x=0;
			r.y=0;
			r.height = 240;*/
			/////////////////////////////////////////////////////////////////////////////////


			Mat img2;
			img1.copyTo(img2);
		cv::rectangle(img2, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
		Mat humanimg(img2,cv::Rect(r.x,r.y,r.width,r.height));

	
	//切り出した人物領域のカラー画像
   
	

   
   //////////////////////////////////////////////////////////////////////////////*/
   //
   //
   //    #face detection
   //
   //
   //
   /////////////////////////////////////////////////////////////////////////////




	char imagename2[10]="image.jpg";

	cv::Mat faceimg(humanimg);
	//imshow("face window",faceimg);
//	imwrite("facedetection.jpg",faceimg);

	//sprintf(imagesample,"sample%d.jpg",i+1); 
	//cv::Mat img = cv::imread(imagesample, 1);
  if(faceimg.empty()){
	printf("cant read img\n");
	  return -1; 
  }
  
  
 // printf("i =%d",i);
  double scale = 2.0;
  cv::Mat gray, smallImg(cv::saturate_cast<int>(faceimg.rows/scale), cv::saturate_cast<int>(faceimg.cols/scale), CV_8UC1);
  // グレースケール画像に変換
  cv::cvtColor(faceimg, gray, CV_BGR2GRAY);
  // 処理時間短縮のために画像を縮小
  cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
  cv::equalizeHist( smallImg, smallImg);
  
  // 分類器の読み込み// Haar-like
  //顔検出
  //  std::string cascadeName = "haarcascade_frontalface_alt2.xml"; 


  //頭部と肩検出
   std::string cascadeName = "HS.xml"; 
  // std::string cascadeName = "./lbpcascade_frontalface.xml"; // LBP
  cv::CascadeClassifier cascade;
  // printf("test");

  if(!cascade.load(cascadeName)){
   printf("can't read\n");
	  return -1;
  }


  std::vector<cv::Rect> faces;
  /// マルチスケール（顔）探索xo
  // 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
  cascade.detectMultiScale(smallImg, faces,
                           1.2, 1,
                           CV_HAAR_SCALE_IMAGE,
                           cv::Size(20, 20));

  
  std::vector<cv::Rect>::const_iterator rface = faces.begin();
  
  
   cv::Point facecenter;
   cv::Point centerRec1,centerRec2;
   FILE *fout;
   if(swt==0){
	   
   fout=fopen(headcoorfile,"a+");

   if(fout==NULL){
   printf("c");
   break;
   }
   }
   cout<<"facessize="<<faces.size()<<endl;


  int radius,facenum=0;

  for(; rface != faces.end(); ++rface) {
 
    facecenter.x = cv::saturate_cast<int>((rface->x + rface->width*0.5)*scale);
    facecenter.y = cv::saturate_cast<int>((rface->y + rface->height*0.5)*scale);
    radius = cv::saturate_cast<int>((rface->width + rface->height)*0.25*scale);
	centerRec1.x=facecenter.x+radius;
	centerRec1.y= facecenter.y+radius;
	centerRec2.x= facecenter.x-radius;
	centerRec2.y= facecenter.y-radius;

	int abc =(int)((centerRec1.x-centerRec2.x));
	int def =(int)((centerRec1.y-centerRec2.y));
	int hij =(int)((centerRec1.x+7*centerRec2.x)*0.125);
	int klm =(int)((centerRec1.y+7*centerRec2.y)*0.125);
	int nop =(int)(cv::saturate_cast<int>(faceimg.rows)*0.7);




	//printf("%d\n",abc);
	//Mat f(img,cv::Rect(facecenter.x,facecenter.y,abc,def));
//	cv::rectangle( faceimg, centerRec1,centerRec2,cv::Scalar(80,80,255), 3, 8, 0 );
//	cv::circle(faceimg,Point(facecenter.x,facecenter.y),8,1,2);
//	cv::circle(faceimg,Point(facecenter.x,r.height/2),8,1,2);



    char imagename[30];

	if(cv::saturate_cast<int>(facecenter.y)<cv::saturate_cast<float>(r.height/2)){
		//cout<<"facecoor="<<cv::saturate_cast<int>(facecenter.y)<<endl;
		if(swt==0){
	fprintf(fout,"%d %d %d %d %d %d\n",framenum,facenum, (int)(centerRec2.x) ,(int)(centerRec2.y) ,abc, def);
		}
	//sprintf(imagename,"./image/%06d.jpg",framenum);
	//imwrite(imagename,faceimg);
	

	
	
	facenum++;
	
			if(facenum==1){
			sprintf(imagename,"./image/%06d.jpg",framenum);
			imwrite(imagename,faceimg);	
			}

	}
  }


  if(swt==0){
	  fclose(fout);
  
  }
  //centerRec2.y= center.y-radius;
// cv::Mat roi_img(humanimg,cv::Rect(centerRec2.x,centerRec2.y,radius*2,radius*2));
// cv::Mat roi_img(img,cv::Rect(100,100,100,3));
  
 humannum++;
 
	//人検出を行う場合
	cv::cvtColor(imgprev, prev1, CV_BGR2GRAY);
	//検出された部分で同じ部分を統合する
	cv::cvtColor(img, next1, CV_BGR2GRAY);
	
	cv::cvtColor(imgprev2, prev22, CV_BGR2GRAY);

	cv::cvtColor(imgprev3, prev33, CV_BGR2GRAY);

	//複数の検出対象がある場合、prevに同じ画像が入る可能性がありそう
	//切り取り、サイズを変更する,人検出を行う場合
	Mat prev(prev1,cv::Rect(r.x,r.y,r.width,r.height));	
	Mat next(next1,cv::Rect(r.x,r.y,r.width,r.height));
	Mat prev2(prev22,cv::Rect(r.x,r.y,r.width,r.height));	
	Mat prev3(prev33,cv::Rect(r.x,r.y,r.width,r.height));	
	
	


	//人検出を行わない場合
	//Mat prev,next;	
	//cv::cvtColor(imgprev, prev, CV_BGR2GRAY);
	//検出された部分で同じ部分を統合する
	//cv::cvtColor(img, next, CV_BGR2GRAY);






	////////////////////////スイッチ///////////

	if(!swt==0){
	



		//読み込みフレーム番号が一致かどうか判定するループ




	//検出した特徴点座標を格納する動的メモリ配列宣言
    std::vector<Point2f> next_pts;
	std::vector<Point2f> prev_pts;
	std::vector<Point2f> prev_pts2;
	std::vector<Point2f> prev_pts3;


	

	//オプティカルフローを計算する領域の中央座標を定義
	//前の画像の大きさに依存する（つまり、連続で処理するならここを工夫する）
	cv::Point2f center = cv::Point(prev.cols/2., prev.rows/2.);
	cv::Point2f center1 = cv::Point(prev2.cols/2., prev2.rows/2.);
	cv::Point2f center2 = cv::Point(prev3.cols/2., prev3.rows/2.);



  for(i=0; i<flowSize.width; ++i) {
    for(int j=0; j<flowSize.width; ++j) {
      cv::Point2f p(i*float(prev.cols)/(flowSize.width-1), 
      j*float(prev.rows)/(flowSize.height-1));

	  cv::Point2f p1(i*float(prev2.cols)/(flowSize.width-1), 
      j*float(prev2.rows)/(flowSize.height-1));

	cv::Point2f p2(i*float(prev3.cols)/(flowSize.width-1), 
      j*float(prev3.rows)/(flowSize.height-1));
      //検出領域の指定
	  prev_pts.push_back((p-center)*0.9f+center);//prev_ptsの末尾に要素を追加
	  prev_pts2.push_back((p-center1)*0.9f+center1);//prev_ptsの末尾に要素を追加
      prev_pts3.push_back((p-center2)*0.9f+center2);//prev_ptsの末尾に要素を追加


    }
  }

  cv::Mat opt(prev);
	
  // Lucas-Kanadeメソッド＋画像ピラミッドに基づくオプティカルフロー
  // parameters=default
#if OPENCV_VERSION_CODE > OPENCV_VERSION(2,3,0)
  cv::Mat status, error;
#else
  std::vector<uchar> status;
  std::vector<float> error;
#endif
  //cv::calcOpticalFlowPyrLK(prev, next, prev_pts, next_pts, status, error);//前画像、後画像、前画像キーポイント、後画像キーポイント
  
  cv::Mat flow(next.size(),CV_32F);
  cv::Mat flow1(prev.size(),CV_32F);
  cv::Mat flow2(prev2.size(),CV_32F);
  
  //cv::calcOpticalFlowFarneback(prev, next, flow,pyrScale,level,winsize,iterations,polyN,polySigma,OPTFLOW_USE_INITIAL_FLOW );
  
 //if((l%10==0)||(l==1)){
  cv::calcOpticalFlowFarneback(prev, next, flow, pyrScale , level, winsize, iterations, polyN, polySigma,OPTFLOW_FARNEBACK_GAUSSIAN  );
 //}
 cv::calcOpticalFlowFarneback(prev2, prev, flow1, pyrScale , level, winsize, iterations, polyN, polySigma,OPTFLOW_FARNEBACK_GAUSSIAN  );
 
 cv::calcOpticalFlowFarneback(prev3, prev2, flow2, pyrScale , level, winsize, iterations, polyN, polySigma,OPTFLOW_FARNEBACK_GAUSSIAN  );
  
	///////////////////////////////////////////////////////////////////////////*/
  

  //過去の画像のOF取得
  //cv::calcOpticalFlowFarneback(prev, next, flow, pyrScale , level, winsize, iterations, polyN, polySigma,OPTFLOW_FARNEBACK_GAUSSIAN  );
 
 





    

	// FILE *fp;  ////もしかしたらここにないとダメぽ
	 float x1,x2,y1,y2;
	 float uleft_x1,uleft_y1,uleft_x2,uleft_y2,uright_x1,uright_y1,uright_x2,uright_y2,lright_x1,lright_y1,lright_x2,lright_y2,lleft_x1,lleft_y1,lleft_x2,lleft_y2;
	
	//char line[255];
	 char row[24];
	// ifstream ifs;
	 //ifs.open("C:\\Users\\mitsuhashi\\Desktop\\lab\\program\\pose_estimation_code_release_v1.21\\example_data\\imgposecoordinates.txt",ios::in);
	
	 
	

	 //腕の座標の読み込み,逐次インタラクションの判定

 for(int beam=0;beam<facenum;beam++){
		
	
		
	
	fscanf(fp,"%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n"
		,&frame,&uleft_x1,&uleft_y1,&uleft_x2,&uleft_y2,&uright_x1,&uright_y1,&uright_x2,&uright_y2,&lleft_x1,&lleft_y1,&lleft_x2,&lleft_y2,&lright_x1,&lright_y1,&lright_x2,&lright_y2);
		
	
	printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",uleft_x1,uleft_y1,uleft_x2,uleft_y2,uright_x1,uright_y1,uright_x2,uright_y2,lleft_x1,lleft_y1,lleft_x2,lleft_y2,lright_x1,lright_y1,lright_x2,lright_y2);
	printf("test");
	printf("frame=%d",frame);
	//circle(humanimg,Point((int)uleft_x1,(int)uleft_y1),5,cv::Scalar(30,200,20),6);	
	//circle(humanimg,Point((int)uleft_x2,(int)uleft_y2),5,cv::Scalar(30,200,20),6);	
	//circle(humanimg,Point((int)uright_x1,(int)uright_y1),5,cv::Scalar(30,200,20),6);	
	//circle(humanimg,Point((int)uright_x2,(int)uright_y2),5,cv::Scalar(30,200,20),6);	

	/*lleft_x1=64.244186;
	lleft_y1=323.671875;
	lleft_x2=49.186047;
	lleft_y2= 266.375000;
	lright_x1= 67.255814;
	lright_y1= 227.171875;
	lright_x2=49.186047;
	lright_y2=148.765625;*/
	circle(humanimg,Point(lleft_x1,lleft_y1),3,cv::Scalar(30,200,206),6);
	circle(humanimg,Point((int)lleft_x2,(int)lleft_y2),3,cv::Scalar(30,200,206),6);	
	circle(humanimg,Point((int)lright_x1,(int)lright_y1),3,cv::Scalar(30,200,20),6);
	circle(humanimg,Point((int)lright_x2,(int)lright_y2),3,cv::Scalar(30,200,20),6);

	
	imwrite("left.jpg",humanimg);

	//２箇所のインタラクション検出ループ

	 Mat drawhandoptical(humanimg);
	int handlabel=0;
	for(handlabel=0;handlabel<2;handlabel++){
	

		
	x1=lleft_x2;
	x2=lleft_x1;

	y1=lleft_y2;
	y2=lleft_y1;


	//右手の座標を代入
	if(handlabel>0){
	
	x1=lright_x2;
	x2=lright_x1;
	y1=lright_y2;
	y2=lright_y1;
	printf("right hand");
	}





   //////////////////////////////////////////////////////////////////////////
   //
   //    #make square
   //
   ///////////////////////////////////////////////////////////////////////////
   int rect=20;
   float width=50;
   int height=50;
   

   //矩形を抽出するかの判定
   //・・・

  // int sqx1,sqy1=0;
  

	std::vector<Point2f>::const_iterator p = prev_pts.begin();
	
	
	
  Mat drawsqoptical(img2);
 // cv::Mat optvec(2,prev_pts.size(),CV_32F);
  cv::Mat optvec(prev_pts.size(),2,CV_32F);
  cv::Mat optvec1(prev_pts2.size(),2,CV_32F);
  cv::Mat optvec2(prev_pts3.size(),2,CV_32F);
  


  Mat_<cv::Vec2d> optveccalc,mapoptcalc;


  cv::Mat mapopt(prev_pts.size(),2,CV_32F);
  cv::Mat mapopt1(prev_pts2.size(),2,CV_32F);
  cv::Mat mapopt2(prev_pts2.size(),2,CV_32F);


     int  flownumber=0;
	 vector<int>sqrflownumber;
	//各フロー点の座標を取得
	 k2=0;
	 k3=0;

	// imshow("sqare",drawhandoptical);

	//x1=x1+10;
	//y2=y2+50;


	 //初期化
	  float handwidth=0;
	  float handheight=0;
      float dis3=1000;
	  float dis4=1000;
	  float sqx1,sqy1,sqx2,sqy2,sqx3,sqy3,sqx4,sqy4;


	  width=cv::saturate_cast<float>(humanimg.rows)*0.05;
	  handwidth=cv::saturate_cast<float>(humanimg.rows)*0.1;
	  handheight=cv::saturate_cast<float>(humanimg.cols)*0.25;
	  cout<<"w="<<handwidth<<endl;

  for(; p!=prev_pts.end(); ++p) {


	  
	   //イテレータpのポインタで指定した座標とそれに対応するfxyを足してる
    const cv::Point2f& fxy = flow.at<cv::Point2f>(p->y, p->x);//動きベクトルの座標､オプティカルフロー後の座標を記述できる
	const cv::Point2f& p1fxy = flow1.at<cv::Point2f>(p->y, p->x);
	const cv::Point2f& p2fxy = flow2.at<cv::Point2f>(p->y, p->x);

	//if((opcount%4==0)){
//		if(fxy.x*fxy.x+fxy.y*fxy.y>1&&(opcount%4==0)){

	//青い線をひく
	//cv::line(drawoptical, *p, *p+fxy, cv::Scalar(150,50,30),1);
	
	//cvArrow(drawoptical, *p, *p+fxy, cv::Scalar(150,150,130),0.1);
	
	//ピンクの玉
	//circle(drawoptical,Point(p->x,p->y),0.5,Scalar(239,117,188));

	//}


	
	//printf("mapopt");
		//(((x2-x1)*a+(y1-y2)*b+y1*((x1-x2)-x1*(y1-y2)))/(sqrt((x1-x2)*(x1-x2)*(y1-y2)*(y1-y2)))<10)&&((x1-x2)/sqrt((x1-x2)*(x1-x2)*(y1-y2)*(y1-y2))-(y1-y2)/sqrt((x1-x2)*(x1-x2)*(y1-y2)*(y1-y2))

	float cos=(y2-y1)/sqrt(floor((x1-x2)*(x1-x2)*1000)/1000+floor(((y1-y2)*(y1-y2))*1000)/1000);
	float sin=(x2-x1)/sqrt(floor((x1-x2)*(x1-x2)*1000)/1000+floor(((y1-y2)*(y1-y2))*1000)/1000);
		
	//人物領域の画素のメモリを割り当てたhumaimgにのみ描画	
	if(handlabel==0){
	line(humanimg, *p, *p+fxy, cv::Scalar(150,50,30),2);
	}

	//a,bは仮
	//腕の矩形領域の判定
	if(((y1<=sin*(p->x-x1)+cos*(p->y-y1)+y1)&&(sin*(p->x-x1)+cos*(p->y-y1)+y1 <= sin*(x2-x1)+cos*(y2-y1)+y1))&&
		(x1-width<=cos*(p->x-x1)-sin*(p->y-y1)+x1)&&(cos*(p->x-x1)-sin*(p->y-y1)+x1<=x1+ width)){//xの範囲も決めないとダメ



	optvec.cv::Mat::row(k2).cv::Mat::col(0)=cv::saturate_cast<float>(fxy.x);
	optvec.cv::Mat::row(k2).cv::Mat::col(1)=cv::saturate_cast<float>(fxy.y);
	
	optvec1.cv::Mat::row(k2).cv::Mat::col(0)=cv::saturate_cast<float>(p1fxy.x);
	optvec1.cv::Mat::row(k2).cv::Mat::col(1)=cv::saturate_cast<float>(p1fxy.y);
	
	optvec2.cv::Mat::row(k2).cv::Mat::col(0)=cv::saturate_cast<float>(p2fxy.x);
	optvec2.cv::Mat::row(k2).cv::Mat::col(1)=cv::saturate_cast<float>(p2fxy.y);
	




	//列の変更
	line(drawhandoptical, *p, *p+fxy, cv::Scalar(250,212,58),2);
	k2++;
	}
	
	//腕の先端のy座標より大&&heightより小さいy座標&&x座標は-5~5までの範囲
			if((sin*(int)(x2-x1)+cos*(int)(y2-y1)+(int)y1<sin*(int)(p->x-x1)+cos*(int)(p->y-y1)+(int)y1)&&(sin*(int)(p->x-x1)+cos*(int)(p->y-y1)+(int)y1 < sin*(int)(x2-x1)+cos*(int)(y2-y1)+(int)y1+(int)handheight)&&
			(cos*(int)(x2-x1)-sin*(int)(y2-y1)+(int)x1-(int)handwidth<cos*(int)(p->x-x1)-sin*(int)(p->y-y1)+(int)x1)&&(cos*(p->x-x1)-sin*(p->y-y1)+x1<cos*(int)(x2-x1)-sin*(int)(y2-y1)+(int)x1+ (int)handwidth)){
	
		
			//同様に矩形領域のOFをmapoptに格納
			mapopt.cv::Mat::row(k3).cv::Mat::col(0)=cv::saturate_cast<float>(fxy.x);
			mapopt.cv::Mat::row(k3).cv::Mat::col(1)=cv::saturate_cast<float>(fxy.y);

			//ひとつ前のOFを取得
			mapopt1.cv::Mat::row(k3).cv::Mat::col(0)=cv::saturate_cast<float>(p1fxy.x);
			mapopt1.cv::Mat::row(k3).cv::Mat::col(1)=cv::saturate_cast<float>(p1fxy.y);
	
			//二つ前のOFを取得
			mapopt2.cv::Mat::row(k3).cv::Mat::col(0)=cv::saturate_cast<float>(p2fxy.x);
			mapopt2.cv::Mat::row(k3).cv::Mat::col(1)=cv::saturate_cast<float>(p2fxy.y);

		    line(drawhandoptical, *p, *p+fxy, cv::Scalar(80,80,255),2);

			k3++;
		
			//矩形領域外形を円で描画
			//circle(drawhandoptical,Point(p->x,p->y),4,cv::Scalar(150,212,58),2);


	//矩形の表示		
	sqx1=cos*(x2-(int)handwidth-x2)+x2;
	sqy1=-sin*(x2-(int)handwidth-x2)+y2;

	sqx2=cos*(x2+(int)handwidth-x2)+x2;
	sqy2=-sin*(x2+(int)handwidth-x2)+y2;

	sqx3=cos*(x2-(int)handwidth-x2)+sin*(int)handheight+x2;
	sqy3=-sin*(x2-(int)handwidth-x2)+y2+(int)handheight;

	
	sqx4=cos*(x2+(int)handwidth-x2)+sin*(int)handheight+x2;
	sqy4=-sin*(x2+(int)handwidth-x2)+(int)handheight+y2;


		
			//フローの添え字を格納
			sqrflownumber.push_back(flownumber);
			}

		flownumber++;

  }


  //矩形描画
  line(drawhandoptical,Point(sqx1,sqy1),Point(sqx2,sqy2),cv::Scalar(200,100,200),2);
  line(drawhandoptical,Point(sqx3,sqy3),Point(sqx4,sqy4),cv::Scalar(200,100,200),2);
  line(drawhandoptical,Point(sqx3,sqy3),Point(sqx1,sqy1),cv::Scalar(200,100,200),2);
  line(drawhandoptical,Point(sqx2,sqy2),Point(sqx4,sqy4),cv::Scalar(200,100,200),2);





 // imshow("drawhandoptical",drawhandoptical);
 // imshow("imgopt",img);



  //imshow("drawspoptical",drawsqoptical);
  imwrite("sq.jpg",drawhandoptical);


  if(handlabel>0){
 // imwrite("hand.jpg",drawsqoptical);
  printf("r\n");
  }
   //////////////////////////////////////////////////////////////////////////
   //#
   //#    #calcurate cos similarlity 
   //#    //前腕と矩形領域の類似度の計算
   //#
   ///////////////////////////////////////////////////////////////////////////
 // cout<<mapopt.row(4)<<endl;
 // cout<<optvec.row(1)<<endl;
 // cout<<mapopt.row(4).dot(optvec.row(1))<<endl;


  cv::Vec2d amat(2,1);
  cv::Vec2d bmat(-2,7);
  
  int sumcosnumber;
  float sumcosmax=-1000;
  float sumcosmax1=-1000;
  float sumcosmax2=-1000;
  float scalarsum=0;
  float scalarsum1=0;
  float scalarsum2=0;
  float cosmean=0;
  float sumcos,sumcos1,sumcos2;


  //矩形領域内のフローが持つcos類似度
  	fout=fopen("simcalcresult_not.txt","a+");
  vector<float> cosall;
  vector<float> cosall1;
  vector<float> cosall2;

		 Mat sumcosall(mapopt.size(),1,CV_32F);
		  for(int j=0;j<k3;j++){

			   sumcos=0;
			   sumcos1=0;
			   sumcos2=0;


			  for(i=0;i<k2;i++){
				  //mat→vec形式に変更して内積計算
				  cv::Vec2d optv(optvec.at<float>(i,0),optvec.at<float>(i,1));
				  cv::Vec2d mapv(mapopt.at<float>(j,0),mapopt.at<float>(j,1));

				  cv::Vec2d optv1(optvec1.at<float>(i,0),optvec1.at<float>(i,1));
				  cv::Vec2d mapv1(mapopt1.at<float>(j,0),mapopt1.at<float>(j,1));

				  cv::Vec2d optv2(optvec2.at<float>(i,0),optvec2.at<float>(i,1));
				  cv::Vec2d mapv2(mapopt2.at<float>(j,0),mapopt2.at<float>(j,1));


	             sumcos=sumcos+(optv.dot(mapv))/(sqrt(optv.dot(optv))*sqrt(mapv.dot(mapv)));//腕のOFと矩形内の内積の和を計算
				 scalarsum=scalarsum+sqrt(optv.dot(optv));
				  

				   sumcos1=sumcos1+(optv1.dot(mapv1))/(sqrt(optv1.dot(optv1))*sqrt(mapv1.dot(mapv1)));//腕のOFと矩形内の内積の和を計算
				 scalarsum1=scalarsum1+sqrt(optv1.dot(optv1));

				 sumcos2=sumcos2+(optv2.dot(mapv2))/(sqrt(optv2.dot(optv2))*sqrt(mapv2.dot(mapv2)));//腕のOFと矩形内の内積の和を計算
				 scalarsum2=scalarsum2+sqrt(optv2.dot(optv2));


				//  cout<<"sumcos=" <<sumcos<<"::k3="<<j<<endl;
		Mat tem1(optv);       
		Mat tem2(mapv);
    
			tem1.release();
			tem2.release();

			  }

	
			  		 
	cosmean=sumcos/k2;


	fprintf(fout,"%f\n",cosmean);

//	
	
			  //ある矩形内の点の内積和を取得
			  sumcosall.cv::Mat::row(j).cv::Mat::col(0)=sumcos;
	

		//cosall＝ある矩形のOFにおける腕領域のOF内積の和
		cosall.push_back(sumcos);
		cosall1.push_back(sumcos1);
		cosall2.push_back(sumcos2);



		//最大値の算出
		float tem3;
	//	tem3=cosall.at(j);
		//printf("tem3=%f\n",tem3);


		//sumcosmaxに代入
		cosall=cosall1;

		//cosallに代入



		//最大値をとるフローナンバと大きさを取得
			if(cv::saturate_cast<float>(cosall.at(j))>sumcosmax){
			//	sumcosnumber=k3;				
				sumcosmax=cv::saturate_cast<float>(cosall.at(j));
			} 

			if(cv::saturate_cast<float>(cosall1.at(j))>sumcosmax1){
		//		sumcosnumber=k3;				
				sumcosmax1=cv::saturate_cast<float>(cosall1.at(j));
			} 

			if(cv::saturate_cast<float>(cosall2.at(j))>sumcosmax2){
		//		sumcosnumber=k3;				
				sumcosmax2=cv::saturate_cast<float>(cosall2.at(j));
			} 



		  }
    

		fclose(fout);
	  

	

	//sqrflownumbe=オプティカルフローのナンバ
		  
		  //イテレータ、フローのカウンタ初期化
	p=prev_pts.begin();
	flownumber=0;
	k3=0;
	vector<int>cosflow;
    	
	

	float gx=0,gy=0,cossimall=0;
	float gx1=0,gy1=0,cossimall1=0;
	float gx2=0,gy2=0,cossimall2=0;

	int maxflownum,maxflownum1,maxflownum2=0;
	//閾値より大きい類似度を持つか判定
	for(;p!=prev_pts.end();p++){
		prev_pts.size();
	//	cout<<"k3="<<k3<<endl;
	//	cout<<"flownumber="<<flownumber<<endl;
	//	cout<<"flowsize="<<prev_pts.size()<<endl;
	//	cout<<"sqrflownumber="<<sqrflownumber.size()<<endl;

		if((sqrflownumber.empty())||(flownumber>cv::saturate_cast<float>(sqrflownumber.back())))
		{
			break;
		}

		else if(flownumber<=cv::saturate_cast<float>(sqrflownumber.back())){
			int y=cv::saturate_cast<int>(sqrflownumber.at(k3));
		//	int last=cv::saturate_cast<int>(sqrflownumber.back());
		//	printf("y=%d",y);
					if((flownumber==y)&&(k3<=sqrflownumber.size())){
		
						//	cout<<"gggg"<<endl;

							if(cv::saturate_cast<float>(cosall.at(k3))/sumcosmax>0.5){
								//cout<<cv::saturate_cast<float>(cosall.at(k3))/sumcosmax)<<endl;
								//cosflow.push_back(k3);
								circle(drawhandoptical,Point(p->x,p->y),1,cv::Scalar(30,200,206),2);
								//line(drawhandoptical,Point(p->x,p->y),2,cv::Scalar(30,200,206),2);
	
								//cosflowに類似度0.5以上になるフローナンバを格納	

							 cossimall=cossimall+cv::saturate_cast<float>(cosall.at(k3));
							 gx = gx + cv::saturate_cast<float>(p->x);
							 gy = gy + cv::saturate_cast<float>(p->y);
	 						 imwrite("intaraction.jpg",drawhandoptical);
							 maxflownum++;
						
							}

							if(cv::saturate_cast<float>(cosall1.at(k3))/sumcosmax1>0.5){
							
							//	circle(drawhandoptical,Point(p->x,p->y),1,cv::Scalar(30,200,206),2);
						
								//cosflowに類似度0.5以上になるフローナンバを格納	

							 cossimall1=cossimall1+cv::saturate_cast<float>(cosall1.at(k3));
							 gx1 = gx1 + cv::saturate_cast<float>(p->x);
							 gy1 = gy1 + cv::saturate_cast<float>(p->y);
	 						// imwrite("intaraction.jpg",drawhandoptical);
							 maxflownum1++;
						
							}

							if(cv::saturate_cast<float>(cosall2.at(k3))/sumcosmax2>0.5){
						//	circle(drawhandoptical,Point(p->x,p->y),1,cv::Scalar(30,200,206),2);
								//cosflowに類似度0.5以上になるフローナンバを格納	

							 cossimall2=cossimall2+cv::saturate_cast<float>(cosall2.at(k3));
							 gx2 = gx2 + cv::saturate_cast<float>(p->x);
							 gy2 = gy2 + cv::saturate_cast<float>(p->y);
//	 						 imwrite("intaraction.jpg",drawhandoptical);
							 maxflownum2++;
						
							}






					//else{
					//	cout<<cosall.at(k3)<<endl;
					//}	
					//imshow("ruizido",drawhandoptical);
					k3++;
					}

		}


		

		flownumber++;
		//}
	//	else{
	//	cout<<"break"<<endl;
	//	break;
		//}
		//エッジの抽出（曖昧に境界をひくのもあり？）
	}

	/////////////////////関数化したい
	

	
   //////////////////////////////////////////////////////////////////////////
   //#
   //#    #特徴量の抽出
   //#
   ///////////////////////////////////////////////////////////////////////////

	



	//重心算出
	if(maxflownum>0){
	gx=gx/maxflownum;
	gy=gy/maxflownum;
	}

	if(maxflownum1>0){
	gx1=gx1/maxflownum1;
	gy1=gy1/maxflownum1;
	}

	if(maxflownum2>0){
	gx2=gx2/maxflownum2;
	gy2=gy2/maxflownum2;
	}



	//特徴量群
	//重心と腕の距離
	float distance=0;
	float framedistance1=0;
	float framedistance2=0;
	float theta;
	
	double pseq;

	
	//vector< float* > array(size1, new float[size2]);

	//array.push_back(new int[size2]);
	//array.push_back(new int[1]);


    float rov_frame_d,rov_theta=0;


	distance=sqrt(floor(gx-x1)*floor(gx-x1)+floor(gy-y1)*floor(gy-y1));
	framedistance1=sqrt(floor(gx1-gx)*floor(gx1-gx)+floor(gy1-gy)*floor(gy1-gy));
	framedistance2=sqrt(floor(gx2-gx1)*floor(gx2-gx1)+floor(gy2-gy1)*floor(gy2-gy1));
	printf("framedistance1=%f",framedistance1);
	printf("framedistance2=%f",framedistance2);
	rov_frame_d=(floor(framedistance1*10000)/10000)/(floor(framedistance2*10000)/10000);
	printf("rov=%f", rov_frame_d);



	
	//vector<float>::iterator it = seqdata.begin();
	//seqdata.push_back(distance);	
	//it=seqdata.erase(it);
	
	//pseq=testHMM(gx,gy,seqdata);
	theta=floor(atan2((gy-y1),(gx-x1))*1000)/1000;
	
	//各人物の腕のフレーム情報　

	printf(" theta=%3f distance= %3f gx = %3f gy= %3f\n",theta,distance,gx,gy);
	//	printf("get");
	svmnum++;
	//	array.push_back(new float[distance]);

	
   //////////////////////////////////////////////////////////////////////////
   //#
   //#    #frame sequence data
   //#
   ///////////////////////////////////////////////////////////////////////////



	//t-1時点のオプティカルフローの座標を現在のOFとして格納
	//prev1_pts=prev_pts;
	//t-2時点のオプティカルフローの座標をt-1のOFとして格納
	//


	//arr.resize(20);		// ()内の数字が要素数になる




		//サイズの動的な拡張
	arr.resize(svmnum);
	//for( int i=0; i<6; i++ ){
	arr[svmnum-1].resize(5);
	//}

		arr[svmnum-1][0]=distance;
		arr[svmnum-1][1]=theta;

		if(sumcosmax==0){
		arr[svmnum-1][2]=0;
		}
    	else{
		arr[svmnum-1][2]=floor(cossimall/sumcosmax*10000)/10000;
		}
		
		//cossimall/sumcosmax;//類似度の総和を正規化したもの(類似度)
		arr[svmnum-1][3]=maxflownum;//OFが一定類似度を超えた数を正規化した数
		arr[svmnum-1][4]=scalarsum;//OFスカラーの差
		arr[svmnum-1][5]=rov_frame_d;
 
	//mapopt.release();
	optvec.release();
	

//	if(frame==5927){
//	break;
 //   }

//	drawsqoptical.release();

	}//腕のインタラクション判定ループ

	drawhandoptical.release();
	humanimg.release();
	
	

	 }//腕の座標の読み込みループ(頭部検出数ループ)

	


	}//人検出数のループend



	
	}////////////////////////////////////////////////////////////////////スイッチend


	//フローの受け渡し
	//p1flow=flow;
	//p2flow=p1flow;h
	
	
	//次の画像行列のアドレスを渡す
   imgprev3=imgprev2; 
   imgprev2=imgprev;
   imgprev=img;
   
   imwrite("img.jpg",img);
   imwrite("imgprev.jpg",imgprev);
   imwrite("imgprev2.jpg",imgprev2);
   imwrite("imgprev3.jpg",imgprev3);

				
   l++;
	 char k =(char)waitKey(30);//30秒待つ
		    if( k == 27 ) break;	
		//prev_pts=next_pts;

			
			}//whileループend（画像のループ）

	


 }//movienumループend

			 fclose(fp);
			
			 //SVMに学習データを入力
			 //ループをきちんと抜けないとできない
			svm(arr);



}




