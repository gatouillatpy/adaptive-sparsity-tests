#include <cuda_fp16.h>
__device__ float lire(float x) { return x; }
__device__ float lire(__half x) { return __half2float(x); }
__device__ float act_gpu(float x,int mode) {
    if(mode==1) return fmaxf(0,x);
    if(mode==2) return .5f*x*(1+tanhf(.7978845608f*(x+.044715f*x*x*x)));
    return x;
}
__global__ void appliquer(float* y,int n,int mode) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=act_gpu(y[i],mode);
}
template<typename T>
__global__ void choisi(const T* w,const T* x,const int* ids,float* y,int d,int mode) {
    int ligne=ids[blockIdx.x];
    float v=0;
    for(int j=threadIdx.x;j<d;j+=blockDim.x)
        v+=lire(w[size_t(ligne)*d+j])*lire(x[j]);
    __shared__ float somme[256];
    somme[threadIdx.x]=v;
    __syncthreads();
    for(int stride=128;stride;stride/=2) {
        if(threadIdx.x<stride) somme[threadIdx.x]+=somme[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0) y[ligne]=act_gpu(somme[0],mode);
}
