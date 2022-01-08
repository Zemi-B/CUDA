#include <stdio.h>
#include<cuda_runtime.h>
#include <stdlib.h>

const int X=5716,Y=3731,W=100;
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
        gpu[yid*X+xid]=V/(kaz*2);
    }
   
}

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
    printf("debug%d\n",__LINE__);
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
            if((i*X+j)%300000==0){printf("%d\n",(int)ch);}
        }
    }
    printf("debug%d\n",__LINE__);
    // デバイス(GPU)のメモリ領域確保
    cudaMalloc((void**)&picgpu, sizeof(int)*X*Y);
    cudaMalloc((void**)&gpuwa, sizeof(int)*X*Y);

    // ホスト(CPU)からデバイス(GPU)へ転送
    cudaMemcpy(picgpu, pic, sizeof(int)*X*Y, cudaMemcpyHostToDevice);

    printf("debug%d\n",__LINE__);
    // スレッド数、ブロック数の設定(説明は他のページ)
    dim3 blocks(16,16);
    dim3 threads((X+15)/16,(Y+15)/16);

    // カーネル(GPUの関数)実行
    cudaKernel<<< blocks, threads >>>(picgpu);

    // デバイス(GPU)からホスト(CPU)へ転送
    cudaMemcpy(pic, picgpu, sizeof(int)*X*Y, cudaMemcpyDeviceToHost);
    printf("debug%d\n",__LINE__);
    
    for (int i = 0; i+W < Y; ++i) {
        for (int j = 0; j+W < X; ++j) {
            fputc(pic[i*X+j], fpout);
            if((i*X+j)%300000==0){printf("%d\n",(int)pic[i*X+j]);}
        }
    }
    printf("debug%d\n",__LINE__);
    fclose(fpin);
    fclose(fpout);
    // ホストメモリ解放
    free(pic);
    
    // デバイスメモリ解放
    cudaFree(picgpu);
    cudaFree(gpuwa);
    // 終了処理
    //CUT_EXIT(argc, argv);
    return 0;
}