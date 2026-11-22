#include "essai.hpp"
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <immintrin.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

Blas::Blas(int threads) {
#ifdef _WIN32
    dll = LoadLibraryA("libopenblas.dll");
    auto symbol = [&](const char* s) { return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(dll), s)); };
#else
    dll = dlopen("libopenblas.so.0", RTLD_NOW);
    if (!dll) dll = dlopen("libopenblas.so", RTLD_NOW);
    auto symbol = [&](const char* s) { return dlsym(dll, s); };
#endif
    if (!dll) throw std::runtime_error("OpenBLAS introuvable (PATH / LD_LIBRARY_PATH)");
    gemv = reinterpret_cast<Gemv>(symbol("cblas_sgemv"));
    gemm = reinterpret_cast<Gemm>(symbol("cblas_sgemm"));
    auto set_threads = reinterpret_cast<void (*)(int)>(symbol("openblas_set_num_threads"));
    auto config = reinterpret_cast<const char* (*)()>(symbol("openblas_get_config"));
    if (!gemv || !gemm || !set_threads || !config)
        throw std::runtime_error("OpenBLAS: missing CBLAS/config symbols");
    set_threads(threads);
    description = config();
}
Blas::~Blas() {
#ifdef _WIN32
    if (dll) FreeLibrary(static_cast<HMODULE>(dll));
#else
    if (dll) dlclose(dll);
#endif
}
float produit(const float* a, const float* b, int n) {
    int j = 0;
    float total = 0;
#ifdef __AVX2__
    __m256 v = _mm256_setzero_ps();
    for (; j + 8 <= n; j += 8)
        v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_loadu_ps(a+j), _mm256_loadu_ps(b+j)));
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    for (float x : tmp) total += x;
#endif
    for (; j < n; ++j) total += a[j] * b[j];
    return total;
}
float activation(float x, const std::string& type) {
    if (type == "relu") return std::max(0.0f, x);
    if (type == "gelu") return 0.5f*x*(1+std::tanh(0.7978845608f*(x+0.044715f*x*x*x)));
    return x;
}
void dense_cpu(Blas& blas, const std::vector<float>& w, const float* x,
               std::vector<float>& y, int d, const std::string& act) {
    blas.gemv(101,111,static_cast<int>(y.size()),d,1,w.data(),d,x,1,0,y.data(),1);
    for (float& a : y) a = activation(a, act);
}
void sparse_cpu(const std::vector<float>& w, const float* x, const std::vector<int>& ids,
                std::vector<float>& y, int d, const std::string& act, int threads) {
    std::fill(y.begin(), y.end(), 0.0f);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
        int r = ids[i];
        y[r] = activation(produit(w.data()+size_t(r)*d, x, d), act);
    }
}
Lsh::Lsh(Blas& blas, const std::vector<float>& w, int dim, int width,
         int b, int l, unsigned seed) : d(dim),n(width),bits(b),tables(l),compte(n) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    int p = bits*tables;
    plans.resize(size_t(p)*d);
    for (float& v : plans) v = normal(rng);
    std::vector<float> signes(size_t(n)*p);
    blas.gemm(101,111,112,n,p,d,1,w.data(),d,plans.data(),d,0,signes.data(),p);
    buckets.resize(tables);
    for (auto& table : buckets) table.resize(size_t(1)<<bits);
    for (int i = 0; i < n; ++i) {
        for (int t = 0; t < tables; ++t) {
            unsigned h = 0;
            for (int j = 0; j < bits; ++j)
                h |= unsigned(signes[size_t(i)*p+t*bits+j] >= 0) << j;
            buckets[t][h].push_back(i);
        }
    }
}
std::vector<int> Lsh::select(const float* x, int budget, bool opposite, int probes, int& candidats) {
    std::vector<int> ids;
    if (budget == n) {
        ids.resize(n);
        std::iota(ids.begin(), ids.end(), 0);
        candidats = n;
        return ids;
    }
    std::fill(compte.begin(), compte.end(), 0);
    unsigned masque = (1u<<bits)-1;
    for (int t = 0; t < tables; ++t) {
        unsigned h = 0;
        for (int j = 0; j < bits; ++j)
            h |= unsigned(produit(plans.data()+size_t(t*bits+j)*d,x,d)>=0) << j;
        auto ajout = [&](unsigned code) {
            for (int row : buckets[t][code]) {
                if (compte[row]++ == 0) ids.push_back(row);
            }
        };
        ajout(h);
        if (opposite) ajout(h^masque);
        if (probes) {
            for (int j = 0; j < bits; ++j) {
                ajout(h^(1u<<j));
                if (opposite) ajout((h^masque)^(1u<<j));
            }
        }
    }
    candidats = static_cast<int>(ids.size());
    if (static_cast<int>(ids.size()) > budget) {
        auto ordre = [&](int a, int b) {
            return compte[a] != compte[b] ? compte[a]>compte[b] : a<b;
        };
        std::nth_element(ids.begin(),ids.begin()+budget,ids.end(),ordre);
        ids.resize(budget);
    }
    std::sort(ids.begin(), ids.end()); // garder un peu de localite
    return ids;
}
size_t Lsh::bytes() const {
    size_t s = plans.capacity()*sizeof(float) + compte.capacity()*sizeof(int);
    s += buckets.capacity()*sizeof(buckets[0]);
    for (const auto& t : buckets) {
        s += t.capacity()*sizeof(t[0]);
        for (const auto& b : t) s += b.capacity()*sizeof(int);
    }
    return s;
}
double quantile(std::vector<double> v, double q) {
    if (v.empty()) throw std::runtime_error("empty sample");
    std::sort(v.begin(),v.end());
    return v[static_cast<size_t>(q*(v.size()-1))];
}
Erreur compare(const std::vector<float>& ref, const std::vector<float>& y,
               const std::vector<int>& ids, unsigned seed) {
    double diff=0, norme=0, autre=0, dot=0;
    int n = static_cast<int>(ref.size()), k = static_cast<int>(ids.size());
    std::vector<int> ordre(n), hasard(n);
    std::iota(ordre.begin(), ordre.end(), 0);
    std::iota(hasard.begin(), hasard.end(), 0);
    std::sort(ordre.begin(),ordre.end(),[&](int a,int b) {
        return std::abs(ref[a]) != std::abs(ref[b]) ? std::abs(ref[a])>std::abs(ref[b]) : a<b;
    });
    std::mt19937 rng(seed);
    std::shuffle(hasard.begin(),hasard.end(),rng);
    std::vector<bool> choisi(n,false), oracle(n,false), random(n,false);
    for (int i : ids) choisi[i] = true;
    for (int i=0; i<k; ++i) { oracle[ordre[i]]=true; random[hasard[i]]=true; }
    double perte_oracle=0, perte_random=0, rappel=0;
    for (int i=0; i<n; ++i) {
        double a=ref[i], b=y[i];
        diff+=(a-b)*(a-b); norme+=a*a; autre+=b*b; dot+=a*b;
        if (!oracle[i]) perte_oracle+=a*a;
        else if (choisi[i]) ++rappel;
        if (!random[i]) perte_random+=a*a;
    }
    return {diff/n, norme>0 ? diff/norme : (diff==0?0:1),
        norme>0 && autre>0 ? dot/std::sqrt(norme*autre) : (norme==0 && autre==0?1:0),
        k ? rappel/k : 0, perte_oracle/n, perte_random/n};
}
void tests_cpu(Blas& blas) {
    auto check = [](bool ok, const char* msg) { if(!ok) throw std::runtime_error(msg); };
    const int d=37,n=128;
    std::mt19937 rng(123);
    std::normal_distribution<float> r;
    std::vector<float> w(n*d),x(d),a(n),b(n);
    for (float& v:w) v=r(rng);
    for (float& v:x) v=r(rng);
    Lsh index(blas,w,d,n,4,6,42);
    int candidats=0;
    auto ids=index.select(x.data(),n,false,0,candidats);
    dense_cpu(blas,w,x.data(),a,d,"gelu");
    sparse_cpu(w,x.data(),ids,b,d,"gelu",1);
    check(compare(a,b,ids,1).relative<1e-10,"full selection != dense");
    ids=index.select(w.data()+17*d,16,false,0,candidats);
    check(std::binary_search(ids.begin(),ids.end(),17),"self query not retrieved");
    check(ids.size()<=16,"budget exceeded");
    check(std::adjacent_find(ids.begin(),ids.end())==ids.end(),"duplicate candidates");
    ids.clear();
    sparse_cpu(w,x.data(),ids,b,d,"gelu",1);
    check(std::all_of(b.begin(),b.end(),[](float v){return v==0;}),"empty selection");
    auto err=compare(a,b,ids,1);
    check(std::abs(err.relative-1)<1e-9 && err.cosine==0,"empty error metrics");
    std::vector<float> constant_w(8*d,1), contraire(d,-1);
    Lsh empty(blas,constant_w,d,8,10,1,2);
    ids=empty.select(contraire.data(),2,false,0,candidats);
    check(ids.empty(),"opposite vector should miss exact bucket");
    ids=empty.select(contraire.data(),2,true,0,candidats);
    check(ids.size()==2 && candidats==8,"signed lookup or budget broken");
    std::vector<float> ref{1,2,3,4}, approx{0,0,3,4};
    err=compare(ref,approx,{2,3},42);
    check(std::abs(err.mse-1.25)<1e-12 && err.recall==1,"known error metrics");
    check(std::abs(err.oracle_mse-1.25)<1e-12,"oracle mask");
}
