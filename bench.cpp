#include "essai.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <numeric>

struct Config {
    int d=768,n=3072,queries=24,repeats=3,threads=1,bits=8,tables=16,probes=0;
    unsigned seed=42;
    bool test=false;
    std::string backend="cpu",precision="fp32",dist="gaussian",act="gelu",csv,weights,inputs;
    std::vector<double> keep{0.05,0.10,0.15};
};
Config options(int argc,char** argv) {
    Config c;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--self-test") { c.test=true; continue; }
        if(a=="--help") {
            std::cout << "bench [--preset gpt2|llama] [--backend cpu|gpu] [--precision fp32|fp16]\n"
              "--keep .05,.1,.15 --queries 24 --repeats 3 --seed 42 --threads 1\n"
              "--bits 8 --tables 16 --probes 0|1 --dist gaussian|relu\n"
              "--activation gelu|relu|linear --dim N --width N --csv path\n"
              "--weights raw-fp32-file --inputs raw-fp32-file --self-test\n";
            std::exit(0);
        }
        if(i+1==argc) throw std::runtime_error("missing value: "+a);
        std::string v=argv[++i];
        if(a=="--preset") {
            if(v=="gpt2") { c.d=768; c.n=3072; }
            else if(v=="llama") { c.d=4096; c.n=11008; }
            else throw std::runtime_error("unknown preset");
        } else if(a=="--dim") c.d=std::stoi(v);
        else if(a=="--width") c.n=std::stoi(v);
        else if(a=="--queries") c.queries=std::stoi(v);
        else if(a=="--repeats") c.repeats=std::stoi(v);
        else if(a=="--threads") c.threads=std::stoi(v);
        else if(a=="--bits") c.bits=std::stoi(v);
        else if(a=="--tables") c.tables=std::stoi(v);
        else if(a=="--probes") c.probes=std::stoi(v);
        else if(a=="--seed") c.seed=static_cast<unsigned>(std::stoul(v));
        else if(a=="--backend") c.backend=v;
        else if(a=="--precision") c.precision=v;
        else if(a=="--dist") c.dist=v;
        else if(a=="--activation") c.act=v;
        else if(a=="--csv") c.csv=v;
        else if(a=="--weights") c.weights=v;
        else if(a=="--inputs") c.inputs=v;
        else if(a=="--keep") {
            c.keep.clear();
            std::istringstream s(v);
            while(std::getline(s,v,',')) c.keep.push_back(std::stod(v));
        } else throw std::runtime_error("unknown argument: "+a);
    }
    if(c.d<1 || c.n<1 || c.d>65536 || c.n>65536 || size_t(c.d)*c.n>size_t(512)*1024*1024
       || c.queries<1 || c.queries>4096 || c.repeats<1 || c.repeats>1000
       || c.threads<1 || c.threads>256 || c.bits<1 || c.bits>16 || c.tables<1 || c.tables>256
       || (c.probes!=0 && c.probes!=1) || c.keep.empty())
        throw std::runtime_error("invalid dimensions or sampling parameters");
    if(size_t(c.bits)*c.tables*c.n>size_t(128)*1024*1024)
        throw std::runtime_error("index build workspace exceeds 512 MiB");
    for(double k:c.keep) if(!std::isfinite(k) || k<=0 || k>1)
        throw std::runtime_error("keep must be in (0,1]");
    if(c.backend!="cpu" && c.backend!="gpu") throw std::runtime_error("backend must be cpu|gpu");
    if(c.precision!="fp32" && c.precision!="fp16") throw std::runtime_error("precision must be fp32|fp16");
    if(c.backend=="cpu" && c.precision!="fp32") throw std::runtime_error("CPU path is FP32 only");
    if(c.dist!="gaussian" && c.dist!="relu") throw std::runtime_error("dist must be gaussian|relu");
    if(c.act!="gelu" && c.act!="relu" && c.act!="linear") throw std::runtime_error("invalid activation");
    return c;
}
void read_floats(const std::string& path,std::vector<float>& v) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if(!in || in.tellg()!=static_cast<std::streamoff>(v.size()*sizeof(float)))
        throw std::runtime_error("wrong raw FP32 file size: "+path);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(float));
    if(!in) throw std::runtime_error("read failed: "+path);
    for(float x:v) if(!std::isfinite(x)) throw std::runtime_error("non-finite input");
}
int main(int argc,char** argv) {
    try {
        Config c=options(argc,argv);
        Blas blas(c.threads);
        std::cerr << blas.description << "\n";
        if(c.test) {
            tests_cpu(blas);
#ifdef AVEC_CUDA
            if(c.backend=="gpu") {
                std::vector<float> w(71*37),x(37),a(71),b(71);
                std::mt19937 r(19);
                std::normal_distribution<float> norm;
                for(float& v:w) v=norm(r);
                for(float& v:x) v=norm(r);
                auto gpu=gpu_create(w,37,71,c.precision,c.act);
                gpu->dense(x.data(),a);
                std::vector<int> ids(71);
                std::iota(ids.begin(),ids.end(),0);
                gpu->sparse(x.data(),ids,b);
                if(!std::isfinite(compare(a,b,ids,1).relative) || compare(a,b,ids,1).relative>1e-7)
                    throw std::runtime_error("GPU full selection mismatch");
                ids={0,7,70};
                gpu->sparse(x.data(),ids,b);
                for(int i=0;i<71;++i) {
                    float expected=std::binary_search(ids.begin(),ids.end(),i)?a[i]:0;
                    if(!std::isfinite(b[i]) || std::abs(b[i]-expected)>0.002f*(1+std::abs(expected)))
                        throw std::runtime_error("GPU subset mismatch");
                }
                ids.clear(); gpu->sparse(x.data(),ids,b);
                for(float v:b) if(v!=0) throw std::runtime_error("GPU empty selection");
                std::cerr << gpu->name() << " " << c.precision << ": tests passed\n";
            }
#else
            if(c.backend=="gpu") throw std::runtime_error("CUDA not compiled");
#endif
            std::cerr << "CPU: tests passed\n";
            return 0;
        }
        std::mt19937 rng(c.seed);
        std::normal_distribution<float> normal;
        std::vector<float> w(size_t(c.d)*c.n),x(size_t(c.queries)*c.d);
        float scale=1/std::sqrt(static_cast<float>(c.d));
        if(c.weights.empty()) for(float& v:w) v=normal(rng)*scale;
        else read_floats(c.weights,w);
        if(c.inputs.empty()) {
            for(float& v:x) {
                v=normal(rng);
                if(c.dist=="relu") v=std::max(0.0f,v);
            }
        } else read_floats(c.inputs,x);
        auto debut=Horloge::now();
        Lsh index(blas,w,c.d,c.n,c.bits,c.tables,c.seed+991);
        double build_ms=ms(debut);
        std::unique_ptr<Gpu> gpu;
        if(c.backend=="gpu") {
#ifdef AVEC_CUDA
            gpu=gpu_create(w,c.d,c.n,c.precision,c.act);
            std::cerr << gpu->name() << " " << c.precision << "\n";
#else
            throw std::runtime_error("CUDA not compiled");
#endif
        }
        std::vector<float> y(c.n);
        std::vector<std::vector<float>> reference(c.queries,std::vector<float>(c.n));
        auto dense = [&](const float* input,std::vector<float>& out) {
            auto t=Horloge::now();
            if(gpu) return gpu->dense(input,out);
            dense_cpu(blas,w,input,out,c.d,c.act);
            return ms(t);
        };
        for(int i=0;i<4;++i) dense(x.data(),y);
        std::vector<double> dense_t;
        for(int r=0;r<c.repeats;++r)
            for(int q=0;q<c.queries;++q)
                dense_t.push_back(dense(x.data()+size_t(q)*c.d,reference[q]));
        std::ofstream file;
        if(!c.csv.empty()) {
            file.open(c.csv);
            if(!file) throw std::runtime_error("cannot write csv");
        }
        std::ostream& out=c.csv.empty()?std::cout:file;
        out << "backend,precision,dim,width,distribution,activation,weights,inputs,seed,threads,queries,repeats,bits,tables,probes,keep_budget,kept_mean,kept_min,kept_max,candidates_mean,dense_p50_ms,dense_p95_ms,select_p50_ms,eval_p50_ms,total_p50_ms,total_p95_ms,speedup,mse,relative_mse,cosine,oracle_recall,oracle_mse,random_mse,index_build_ms,index_mib,weights_mib\n";
        out << std::setprecision(9);
        for(double keep:c.keep) {
            int budget=std::max(1,static_cast<int>(std::ceil(keep*c.n))), candidats=0;
            std::vector<double> selection_t,calcul_t,total_t;
            double kept=0,nb_candidats=0; int kept_min=c.n,kept_max=0;
            Erreur err{};
            auto essai = [&](int q,bool mesure,bool quality) {
                const float* input=x.data()+size_t(q)*c.d;
                auto t=Horloge::now();
                auto ids=index.select(input,budget,c.act=="linear",c.probes,candidats);
                double select_ms=ms(t);
                auto calc=Horloge::now();
                double eval_ms=0;
                if(gpu) eval_ms=gpu->sparse(input,ids,y);
                else { sparse_cpu(w,input,ids,y,c.d,c.act,c.threads); eval_ms=ms(calc); }
                double total_ms=ms(t);
                if(mesure) {
                    selection_t.push_back(select_ms);
                    calcul_t.push_back(eval_ms);
                    total_t.push_back(total_ms);
                }
                if(quality) {
                    kept+=double(ids.size())/c.n; nb_candidats+=candidats;
                    kept_min=std::min(kept_min,int(ids.size())); kept_max=std::max(kept_max,int(ids.size()));
                    auto e=compare(reference[q],y,ids,c.seed+q);
                    err.mse+=e.mse; err.relative+=e.relative; err.cosine+=e.cosine;
                    err.recall+=e.recall; err.oracle_mse+=e.oracle_mse; err.random_mse+=e.random_mse;
                }
            };
            for(int i=0;i<4;++i) essai(i%c.queries,false,false);
            for(int r=0;r<c.repeats;++r)
                for(int q=0;q<c.queries;++q) essai(q,true,r==0);
            double dt=quantile(dense_t,.5), st=quantile(total_t,.5), count=c.queries;
            out << c.backend << ',' << c.precision << ',' << c.d << ',' << c.n << ','
                << c.dist << ',' << c.act << ',' << (c.weights.empty()?"synthetic":"file") << ','
                << (c.inputs.empty()?"synthetic":"file") << ',' << c.seed << ',' << c.threads << ','
                << c.queries << ',' << c.repeats << ',' << c.bits << ',' << c.tables << ',' << c.probes << ','
                << keep << ',' << kept/count << ',' << kept_min << ',' << kept_max << ',' << nb_candidats/count << ','
                << dt << ',' << quantile(dense_t,.95) << ',' << quantile(selection_t,.5) << ','
                << quantile(calcul_t,.5) << ',' << st << ',' << quantile(total_t,.95) << ',' << dt/st << ','
                << err.mse/count << ',' << err.relative/count << ',' << err.cosine/count << ','
                << err.recall/count << ',' << err.oracle_mse/count << ',' << err.random_mse/count << ','
                << build_ms << ',' << index.bytes()/1048576.0 << ',' << w.size()*4/1048576.0 << '\n';
            std::cerr << "keep " << keep << " actual " << kept/count << " dense " << dt
                << " sparse " << st << " ms, speedup " << dt/st << ", rel MSE " << err.relative/count << "\n";
        }
        return 0;
    } catch(const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
