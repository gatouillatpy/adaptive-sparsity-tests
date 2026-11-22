#include "essai.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <fstream>
#include <sstream>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <numeric>

static void cuda_ok(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void blas_ok(cublasStatus_t e) {
    if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status "+std::to_string(int(e)));
}
static void driver_ok(CUresult e) {
    if(e!=CUDA_SUCCESS) {
        const char* message=nullptr;
        cuGetErrorString(e,&message);
        throw std::runtime_error(message?message:"CUDA driver error");
    }
}
struct GpuEssai : Gpu {
    int d,n,mode;
    bool half;
    void* poids=nullptr;
    void* entree=nullptr;
    int* indices=nullptr;
    float* sortie=nullptr;
    std::vector<__half> xhalf;
    cublasHandle_t handle=nullptr;
    cudaDeviceProp prop{};
    CUmodule module=nullptr;
    CUfunction act_fn=nullptr, sparse_fn=nullptr;
    void compile() {
        std::ifstream file("gpu.cu");
        if(!file) throw std::runtime_error("Run from the project directory (gpu.cu required)");
        std::ostringstream buffer; buffer << file.rdbuf();
        auto source=buffer.str();
        nvrtcProgram program=nullptr;
        auto nv_ok=[](nvrtcResult r) {
            if(r!=NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(r));
        };
        nv_ok(nvrtcCreateProgram(&program,source.c_str(),"gpu.cu",0,nullptr,nullptr));
        std::string arch="--gpu-architecture=compute_"+std::to_string(prop.major)+std::to_string(prop.minor);
        const char* cuda_path=std::getenv("CUDA_PATH");
        if(!cuda_path) throw std::runtime_error("CUDA_PATH must point to the toolkit");
        std::string include=std::string("--include-path=")+cuda_path+"/include";
        const char* args[]={arch.c_str(),include.c_str(),"--std=c++17"};
        nvrtcAddNameExpression(program,"appliquer");
        const char* sparse_name=half?"choisi<__half>":"choisi<float>";
        nvrtcAddNameExpression(program,sparse_name);
        auto status=nvrtcCompileProgram(program,3,args);
        if(status!=NVRTC_SUCCESS) {
            size_t size=0; nvrtcGetProgramLogSize(program,&size);
            std::string log(size,'\0'); nvrtcGetProgramLog(program,log.data());
            nvrtcDestroyProgram(&program);
            throw std::runtime_error(log);
        }
        const char* lowered=nullptr;
        nv_ok(nvrtcGetLoweredName(program,"appliquer",&lowered));
        std::string act_name=lowered;
        nv_ok(nvrtcGetLoweredName(program,sparse_name,&lowered));
        std::string dot_name=lowered;
        size_t size=0; nv_ok(nvrtcGetPTXSize(program,&size));
        std::vector<char> ptx(size); nv_ok(nvrtcGetPTX(program,ptx.data()));
        nv_ok(nvrtcDestroyProgram(&program));
        driver_ok(cuModuleLoadData(&module,ptx.data()));
        driver_ok(cuModuleGetFunction(&act_fn,module,act_name.c_str()));
        driver_ok(cuModuleGetFunction(&sparse_fn,module,dot_name.c_str()));
    }
    GpuEssai(const std::vector<float>& w,int dim,int width,const std::string& precision,
             const std::string& act):d(dim),n(width),mode(act=="relu"?1:act=="gelu"?2:0),
                                     half(precision=="fp16"),xhalf(d) {
        cuda_ok(cudaGetDeviceProperties(&prop,0));
        blas_ok(cublasCreate(&handle));
        blas_ok(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
        compile();
        size_t taille=half?sizeof(__half):sizeof(float);
        cuda_ok(cudaMalloc(&poids,w.size()*taille));
        cuda_ok(cudaMalloc(&entree,size_t(d)*taille));
        cuda_ok(cudaMalloc(&indices,size_t(n)*sizeof(int)));
        cuda_ok(cudaMalloc(&sortie,size_t(n)*sizeof(float)));
        if(half) {
            std::vector<__half> tmp(w.size());
            for(size_t i=0;i<w.size();++i) tmp[i]=__float2half(w[i]);
            cuda_ok(cudaMemcpy(poids,tmp.data(),tmp.size()*taille,cudaMemcpyHostToDevice));
        } else cuda_ok(cudaMemcpy(poids,w.data(),w.size()*taille,cudaMemcpyHostToDevice));
    }
    ~GpuEssai() override {
        if(module) cuModuleUnload(module);
        if(handle) cublasDestroy(handle);
        cudaFree(poids); cudaFree(entree); cudaFree(indices); cudaFree(sortie);
    }
    std::string name() const override {
        return std::string(prop.name)+" compute "+std::to_string(prop.major)+"."+std::to_string(prop.minor);
    }
    void input(const float* x) {
        if(half) {
            for(int i=0;i<d;++i) xhalf[i]=__float2half(x[i]);
            cuda_ok(cudaMemcpy(entree,xhalf.data(),size_t(d)*sizeof(__half),cudaMemcpyHostToDevice));
        } else cuda_ok(cudaMemcpy(entree,x,size_t(d)*sizeof(float),cudaMemcpyHostToDevice));
    }
    void output(std::vector<float>& y) {
        cuda_ok(cudaGetLastError());
        cuda_ok(cudaMemcpy(y.data(),sortie,size_t(n)*sizeof(float),cudaMemcpyDeviceToHost));
    }
    double dense(const float* x,std::vector<float>& y) override {
        auto t=Horloge::now();
        input(x);
        float one=1,zero=0;
        if(half) {
            blas_ok(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,1,d,&one,
                poids,CUDA_R_16F,d,entree,CUDA_R_16F,d,&zero,sortie,CUDA_R_32F,n,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        } else {
            blas_ok(cublasSgemv(handle,CUBLAS_OP_T,d,n,&one,static_cast<float*>(poids),d,
                static_cast<float*>(entree),1,&zero,sortie,1));
        }
        void* args[]={&sortie,&n,&mode};
        driver_ok(cuLaunchKernel(act_fn,(n+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
        output(y);
        return ms(t);
    }
    double sparse(const float* x,const std::vector<int>& ids,std::vector<float>& y) override {
        auto t=Horloge::now();
        input(x);
        cuda_ok(cudaMemset(sortie,0,size_t(n)*sizeof(float)));
        if(!ids.empty()) {
            cuda_ok(cudaMemcpy(indices,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice));
            void* args[]={&poids,&entree,&indices,&sortie,&d,&mode};
            driver_ok(cuLaunchKernel(sparse_fn,unsigned(ids.size()),1,1,256,1,1,0,nullptr,args,nullptr));
        }
        output(y);
        return ms(t);
    }
};
std::unique_ptr<Gpu> gpu_create(const std::vector<float>& w,int d,int n,
                               const std::string& precision,const std::string& act) {
    return std::make_unique<GpuEssai>(w,d,n,precision,act);
}
