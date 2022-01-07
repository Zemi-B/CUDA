#include <stdio.h>
#include<cuda_runtime.h>

#define N 100

const int X=5716,Y=3731;
// カーネル(GPUの関数)
__global__ void cudaKernel(int gpu[]){
    // スレッドID

    int xid=blockIdx.x*blockDim.x+threadIdx.x;
    int yid=blockIdx.y*blockDim.y+threadIdx.y;
    //50近傍の和を愚直にとる
    int V=0,kaz=0;
    for(int dy=-50;dy<=50;dy++){
        for(int dx=-50;dx<=50;dy++){
            int sx=xid-dx;
            int sy=yid-dy;
            if(sx<0||sy<0||sx>X||sy>Y){continue;}
            V+=gpu[sy*X+sx];
            kaz++;
        }
    }
    __syncthreads();
    if(0<=yid&&yid<Y&&0<=xid&&xid<X){
        gpu[yid*X+xid]=V/kaz;
    }
   
}

int main(int argc, char** argv){
    
    static int pic[Y][X];
    int* picgpu;
    unsigned int * gpuwa;

    //デバイスの初期化
    //CUT_DEVICE_INIT(argc, argv);

    FILE *fpin, *fpout;
    char *fin = argv[1];
    char *fout = argv[2];

    fpin = fopen(fin, "r");
    fpout = fopen(fout, "w");
    // xの初期化
    printf("debug%d\n",__LINE__);
    //xとyをどうするかについて考える
    //bmp形式について考える
    char ch;
    for (int i = 0; i < 3; ++i) {
        while ((ch = fgetc(fpin)) != EOF) {
            if (i != 1)
                fputc(ch, fpout);
            if (ch == '\n')
                break;
        }
        if (i == 1) {
            fprintf(fpout, "%d %d\n", Y, X);
        }
    }

    for (int i = 0; i < Y; ++i) {
        for (int j = 0; j < X; ++j) {
            ch = fgetc(fpin);
            pic[i][j] = ch;
        }
    }
    printf("debug%d\n",__LINE__);
    // デバイス(GPU)のメモリ領域確保
    cudaMalloc((void**)&picgpu, sizeof(int)*X*Y);
    cudaMalloc((void**)&gpuwa, sizeof(int)*X*Y);

    // ホスト(CPU)からデバイス(GPU)へ転送
    cudaMemcpy(pic, picgpu, sizeof(int)*X*Y, cudaMemcpyHostToDevice);

    printf("debug%d\n",__LINE__);
    // スレッド数、ブロック数の設定(説明は他のページ)
    dim3 blocks(16,16);
    dim3 threads((Y+15)/16,(X+15)/16);

    // カーネル(GPUの関数)実行
    //cudaKernel<<< blocks, threads >>>(picgpu);

    // デバイス(GPU)からホスト(CPU)へ転送
    cudaMemcpy(pic, picgpu, sizeof(int)*X*Y, cudaMemcpyDeviceToHost);
    printf("debug%d\n",__LINE__);
    
    for (int i = 0; i < Y; ++i) {
        for (int j = 0; j < X; ++j) {
            fputc(pic[i][j], fpout);
        }
    }
    printf("debug%d\n",__LINE__);
    fclose(fpin);
    fclose(fpout);
    // ホストメモリ解放
    //free(pic);
    
    // デバイスメモリ解放
    cudaFree(picgpu);
    cudaFree(gpuwa);
    // 終了処理
    //CUT_EXIT(argc, argv);
    return 0;
}