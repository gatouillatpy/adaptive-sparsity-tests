#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

using Horloge = std::chrono::steady_clock;
inline double ms(Horloge::time_point t) {
    return std::chrono::duration<double, std::milli>(Horloge::now() - t).count();
}
struct Blas {
    void* dll = nullptr;
    using Gemv = void (*)(int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
    using Gemm = void (*)(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
    Gemv gemv = nullptr;
    Gemm gemm = nullptr;
    explicit Blas(int threads);
    ~Blas();
    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    std::string description;
};
float produit(const float* a, const float* b, int n);
float activation(float x, const std::string& type);
void dense_cpu(Blas&, const std::vector<float>& w, const float* x,
               std::vector<float>& y, int d, const std::string& act);
void sparse_cpu(const std::vector<float>& w, const float* x,
                const std::vector<int>& ids, std::vector<float>& y,
                int d, const std::string& act, int threads);

struct Lsh {
    int d, n, bits, tables;
    std::vector<float> plans;
    std::vector<std::vector<std::vector<int>>> buckets;
    std::vector<int> compte;
    Lsh(Blas&, const std::vector<float>& w, int dim, int width,
        int b, int l, unsigned seed);
    std::vector<int> select(const float* x, int budget, bool opposite, int probes,
                            int& candidats);
    size_t bytes() const;
};

struct Erreur { double mse, relative, cosine, recall, oracle_mse, random_mse; };
Erreur compare(const std::vector<float>& reference, const std::vector<float>& y,
               const std::vector<int>& ids, unsigned seed);
double quantile(std::vector<double> v, double q);
void tests_cpu(Blas&);

struct Gpu {
    virtual ~Gpu() = default;
    virtual double dense(const float*, std::vector<float>&) = 0;
    virtual double sparse(const float*, const std::vector<int>&, std::vector<float>&) = 0;
    virtual std::string name() const = 0;
};
#ifdef AVEC_CUDA
std::unique_ptr<Gpu> gpu_create(const std::vector<float>&, int d, int n,
                              const std::string& precision, const std::string& act);
#endif
