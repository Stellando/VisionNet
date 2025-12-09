#ifndef VN_STRUCTURES_H
#define VN_STRUCTURES_H

#include <vector>
#include <random>
#include <functional>
#include <limits>

using namespace std;

// 定義適應值函數類型
using FitnessFunction = function<double(const vector<double>&)>;

// 點 (Individual/Node)
struct Point {
    int id;                     // 全域索引
    vector<double> position;
    double fitness;
    double Cr;                  // 記錄該次更新使用的參數
    double F;
    
    // 建構子
    Point() : id(-1), fitness(numeric_limits<double>::infinity()), Cr(0.0), F(0.0) {}
    Point(int dim, int idx) : id(idx), position(dim), fitness(numeric_limits<double>::infinity()), Cr(0.0), F(0.0) {}
};

// 網格 (Grid/Region) - 由4個點組成
struct GridRegion {
    int id;
    int pointIndices[4]; // 左上, 右上, 左下, 右下 的點在 points 陣列中的索引
    
    // 期望值相關參數
    int Ia = 1;      // 造訪次數
    int Ib = 1;      // 未造訪次數
    double EI = 0.0;
    double Eavg = 0.0;
    double Ebest = 0.0;
    double expectedValue = 0.0; // 最終期望值
    
    GridRegion() : id(-1) {}
};

// 歷史記憶表管理
class HistoryTable {
public:
    vector<double> MCr_mem;
    vector<double> MF_mem;
    int Hsize;
    int k_idx = 0; // 循環指針

    HistoryTable(int size, double initMCr = 0.5, double initMF = 0.5) : Hsize(size) {
        MCr_mem.resize(Hsize, initMCr); 
        MF_mem.resize(Hsize, initMF);
    }

    // 獲取隨機條目
    pair<double, double> getRandomEntry(mt19937& rng) {
        uniform_int_distribution<int> dist(0, Hsize - 1);
        int idx = dist(rng);
        return {MCr_mem[idx], MF_mem[idx]};
    }
};

// Vision Net 主要參數結構
struct VisionNetParams {
    int L;              // 網子邊長 (點的數量為 L*L)
    int dimension;      // 問題維度
    int maxEvaluations; // 最大評估次數
    int Hsize;          // 歷史記憶表大小
    double initialMCr = 0.5;
    double initialMF = 0.5;
    vector<double> lowerBounds;
    vector<double> upperBounds;
    
    VisionNetParams(int gridSize = 5, int dim = 10) 
        : L(gridSize), dimension(dim), maxEvaluations(dim * 10000), Hsize(10),
          initialMCr(0.5), initialMF(0.5) {
        lowerBounds.resize(dim, -100.0);
        upperBounds.resize(dim, 100.0);
    }
    
    // 設定搜尋範圍
    void setBounds(const vector<double>& lower, const vector<double>& upper) {
        lowerBounds = lower;
        upperBounds = upper;
    }
};

// 常用的適應度函數（最小化問題）
namespace FitnessFunctions {
    double Sphere(const vector<double>& x);        // Sphere 函數
    double Ackley(const vector<double>& x);        // Ackley 函數
    double Rastrigin(const vector<double>& x);     // Rastrigin 函數
    double OneMax(const vector<double>& x);        // OneMax 問題（轉為最小化）
}

#endif 