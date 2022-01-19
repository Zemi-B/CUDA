#include <stdio.h>
#include<cuda_runtime.h>
#include <stdlib.h>
#include<chrono>
#include <vector>
#include <iostream>
const int X=5716,Y=3731,W=50;
// カーネル(GPUの関数)
__global__ void cudaKernel(int *gpu){
    // スレッドID

    int xid=blockIdx.x*blockDim.x+threadIdx.x;
    int yid=blockIdx.y*blockDim.y+threadIdx.y;
    //W近傍の和を愚直にとる
    int V=0,kaz=0;
    for(int dy=0;dy<W;dy++){
        for(int dx=0;dx<W;dx++){
            int sx=xid+dx;
            int sy=yid+dy;
            if(sx<0||sy<0||sx>=X||sy>=Y){continue;}
            V+=gpu[sy*X+sx];
            kaz++;
        }
    }
    __syncthreads();
    if(0<=yid&&yid+W<Y&&0<=xid&&xid+W<X){
        gpu[yid*X+xid]=V/kaz;
    }
   
}
using namespace std;
int main(int argc, char** argv){
    
    //static int pic[Y*X];
    int* pic;
    int* picgpu;
    int * gpuwa;
    
    pic=(int*)malloc(sizeof(int)*X*Y);
    //デバイスの初期化
    //CUT_DEVICE_INIT(argc, argv);

    FILE *fpin, *fpout;
    char *fin = argv[1];
    char *fout = argv[2];

    fpin = fopen(fin, "r");
    fpout = fopen(fout, "w");
    // xの初期化
    //printf("debug%d\n",__LINE__);
    //xとyをどうするかについて考える
    //bmp形式について考える
    unsigned char ch;
    for (int i = 0; i < 3; ++i) {
        while ((ch = fgetc(fpin)) != EOF) {
            if (i != 1){ fputc(ch, fpout);}
            if(i==1){printf("%c",ch);}
            if (ch == '\n'){break;}
        }
        if (i == 1) {
            fprintf(fpout, "%d %d\n", X-W, Y-W);
            printf("%d %d\n", X-W, Y-W);
        }
    }

    for (int i = 0; i < Y; ++i) {
        for (int j = 0; j < X; ++j) {
            ch = fgetc(fpin);
            pic[i*X+j] = ch;
            //if((i*X+j)%300000==0){printf("%d\n",(int)ch);}
        }
    }
    std::cout <<"debug"<<__LINE__<<endl;
    auto startA = std::chrono::system_clock::now(); 
    //printf("debug%d\n",__LINE__);
    // デバイス(GPU)のメモリ領域確保
    cudaMalloc((void**)&picgpu, sizeof(int)*X*Y);
    //cudaMalloc((void**)&gpuwa, sizeof(int)*X*Y);
    std::cout <<"debug"<<__LINE__<<endl;
    // ホスト(CPU)からデバイス(GPU)へ転送
    auto startB = std::chrono::system_clock::now(); 
    cudaMemcpy(picgpu, pic, sizeof(int)*X*Y, cudaMemcpyHostToDevice);
          // 計測終了時刻を保存
    //printf("debug%d\n",__LINE__);
    // スレッド数、ブロック数の設定(説明は他のページ)
    dim3 blocks((X+15)/16,(Y+15)/16);
    dim3 threads(16,16);
    auto start = std::chrono::system_clock::now(); 
    // カーネル(GPUの関数)実行
    cudaKernel<<< blocks, threads >>>(picgpu);
    auto end = std::chrono::system_clock::now(); 
    // デバイス(GPU)からホスト(CPU)へ転送
    cudaMemcpy(pic, picgpu, sizeof(int)*X*Y, cudaMemcpyDeviceToHost);
    //printf("debug%d\n",__LINE__);
    auto endA = std::chrono::system_clock::now(); 
    std::cout <<"debug"<<__LINE__<<endl;
    for (int i = 0; i+W < Y; ++i) {
        for (int j = 0; j+W < X; ++j) {
            fputc(pic[i*X+j], fpout);
            //if((i*X+j)%300000==0){printf("%d\n",(int)pic[i*X+j]);}
        }
    }

    //printf("debug%d\n",__LINE__);

    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 要した時間をミリ秒（1/1000秒）に変換して表示
    std::cout <<"keisan="<< msec << " milli sec \n";
    auto Amsec = std::chrono::duration_cast<std::chrono::milliseconds>(endA-startA).count();
    std::cout <<"All="<< Amsec << " milli sec \n";
    auto Bmsec = std::chrono::duration_cast<std::chrono::milliseconds>(endA-startB).count();
    std::cout <<"notnalloc="<< Bmsec << " milli sec \n";
    fclose(fpin);
    fclose(fpout);
    // ホストメモリ解放
    free(pic);
    
    // デバイスメモリ解放
    cudaFree(picgpu);
    //cudaFree(gpuwa);
    // 終了処理
    //CUT_EXIT(argc, argv);
    return 0;
}