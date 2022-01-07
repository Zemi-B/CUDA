#include <stdio.h>


#define N 100

const int xsize=5716,ysize=3731;
// カーネル(GPUの関数)
__global__ void cudaKernel(int *gpu){
    // スレッドID

    int xid=blockIdx.x*blockDim.x+threadIdx.x;
    int yid=blockIdx.y*blockDim.y+threadIdx.y;
    //50近傍の和を愚直にとる
    int V=0,kaz=0;
    for(dy=-50;dy<=50;dy++){
        for(dx=-50;dx<=50;dy++){
            int sx=xid-dx;
            int sy=yid-dy;
            if(sx<0||sy<0||sx>xsize||sy>ysize){continue;}
            V+=gpu[sy][sx];
            kaz++;
        }
    }
    __syncthreads();
    gpu[yid][xid]=V/kaz;
}

int main(int argc, char** argv){
    
    int pic[ysize][xsize];
    int* picgpu;
    unsigned int * gpuwa;

    //デバイスの初期化
    CUT_DEVICE_INIT(argc, argv);

    // xの初期化

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
            fprintf(fpout, "%d %d\n", Y - W, X - W);
        }
    }

    for (int i = 0; i < Y; ++i) {
        for (int j = 0; j < X; ++j) {
            ch = fgetc(fpin);
            pic[i][j] = ch;
        }
    }

    // デバイス(GPU)のメモリ領域確保
    CUDA_SAFE_CALL(cudaMalloc((void**)&picgpu, sizeof(int)*X*Y));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpuwa, sizeof(int)*X*Y));

    // ホスト(CPU)からデバイス(GPU)へ転送
    CUDA_SAFE_CALL(cudaMemcpy(pic, picgpu, sizeof(int)*X*Y, cudaMemcpyHostToDevice));

    
    // スレッド数、ブロック数の設定(説明は他のページ)
    dim3 blocks(16,16);
    dim3 threads((ysize+15)/16,(xsize+15)/16);

    // カーネル(GPUの関数)実行
    cudaKernel<<< blocks, threads >>>(gpu);

    // デバイス(GPU)からホスト(CPU)へ転送
    CUDA_SAFE_CALL(cudaMemcpy(pic, picgpu, sizeof(int)*X*Y, cudaMemcpyDeviceToHost));

    

    // ホストメモリ解放
    free(pic);
    
    // デバイスメモリ解放
    CUDA_SAFE_CALL(cudaFree(picgpu));
    CUDA_SAFE_CALL(cudaFree(gpuwa));
    // 終了処理
    CUT_EXIT(argc, argv);
    return 0;
}