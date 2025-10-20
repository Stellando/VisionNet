#ifndef VN_STRUCTURES_H
#define VN_STRUCTURES_H

#include <vector>
#include <random>
#include <deque>
#include <memory>
#include <functional>

using namespace std;

// 解決方案結構
struct Solution {
    vector<double> position;    // 解的位置
    double fitness;             // 適應度值
    int dimension;              // 維度
    
    Solution(int dim = 0) : dimension(dim), fitness(0.0) {
        if (dim > 0) {
            position.resize(dim, 0.0);
        }
    }
    
    // 複製建構函數
    Solution(const Solution& other) 
        : position(other.position), fitness(other.fitness), dimension(other.dimension) {}
    
    // 賦值運算符
    Solution& operator=(const Solution& other) {
        if (this != &other) {
            position = other.position;
            fitness = other.fitness;
            dimension = other.dimension;
        }
        return *this;
    }
};

// 歷史記憶表參數結構
struct HistoryParameter {
    double MCr;                 // 交配率
    double MF;                  // 縮放因子
    int dimension;              // 對應的維度
    int generation;             // 產生的世代
    double performance;         // 該參數組合的效能
    
    HistoryParameter(double cr = 0.3, double f = 0.3, int dim = 0, int gen = 0)
        : MCr(cr), MF(f), dimension(dim), generation(gen), performance(0.0) {}
};

// 歷史記憶表項目
struct HistoryEntry {
    Solution solution;          // 歷史解
    double expectedValue;       // 期望值
    int generation;             // 產生的世代
    
    HistoryEntry(const Solution& sol, double expVal, int gen)
        : solution(sol), expectedValue(expVal), generation(gen) {}
};

// 歷史記憶表（重新設計）
class HistoryTable {
private:
    vector<HistoryParameter> parameters;  // 歷史參數表
    int maxSize;                          // Hsize * dimension
    int dimension;                        // 問題維度
    
public:
    HistoryTable(int hsize, int dim) 
        : maxSize(hsize * dim), dimension(dim) {
        parameters.reserve(maxSize);
    }
    
    void addParameter(const HistoryParameter& param);
    HistoryParameter getBestParameter(int dim) const;
    vector<HistoryParameter> getRecentParameters(int count) const;
    void updatePerformance(int index, double performance);
    void clear();
    bool isEmpty() const { return parameters.empty(); }
    int size() const { return parameters.size(); }
    int getSize() const { return parameters.size(); }
    void updateMCr(int index, double newMCr);
    void updateMF(int index, double newMF);
};

// 網格點結構
struct GridPoint {
    vector<double> position;    // 網格點位置
    double fitness;             // 適應度值
    int gridId;                 // 所屬網格區域編號
    double Cr;                  // 交配率（用於DE更新）
    double F;                   // 比例係數（用於DE更新）
    
    GridPoint(int dim = 0, int gId = -1) 
        : fitness(0.0), gridId(gId), Cr(0.0), F(0.0) {
        if (dim > 0) {
            position.resize(dim, 0.0);
        }
    }
};



// 網格區域結構
struct GridRegion {
    int regionId;               // 區域編號
    vector<GridPoint> points;   // 該區域的網格點（4個點）
    vector<GridPoint> previousPoints; // 前一代的網格點（用於計算Eavg）
    double expectedValue;       // 期望值 Ei
    pair<int, int> gridCoord;   // 網格座標 (i, j) 在 L×L 網格中
    
    // 期望值計算相關
    int Ia;                     // 被造訪次數
    int Ib;                     // 未造訪次數
    double EI;                  // 造訪比例 EI,i = Ib / Ia
    double Eavg;                // 平均增進值
    double Ebest;               // 最佳目標值
    
    // 正規化後的值
    double EI_normalized;       // 正規化並反轉後的 EI
    double Eavg_normalized;     // 正規化並反轉後的 Eavg
    double Ebest_normalized;    // 正規化並反轉後的 Ebest
    
    GridRegion(int id = -1) 
        : regionId(id), expectedValue(0.0), gridCoord({-1, -1}),
          Ia(1), Ib(1), EI(0.0), Eavg(0.0), Ebest(0.0),
          EI_normalized(0.0), Eavg_normalized(0.0), Ebest_normalized(0.0) {
        points.reserve(4);           // 每個網格有4個點
        previousPoints.reserve(4);   // 前一代的4個點
    }
};

// Vision Net 主要參數結構
struct VisionNetParams {
    int L;                      // 網格邊長（網格總數 = L×L）
    int maxEvaluations;         // 最大函數評估次數 tmax
    int Hsize;                  // 歷史記憶表大小參數
    int dimension;              // 問題維度
    vector<double> lowerBounds; // 每維度的下界 min(xi)
    vector<double> upperBounds; // 每維度的上界 max(xi)
    double initialMCr;          // 初始交配率
    double initialMF;           // 初始縮放因子
    
    VisionNetParams(int gridSize = 5, int dim = 10) 
        : L(gridSize), maxEvaluations(10000), Hsize(50), dimension(dim),
          initialMCr(0.3), initialMF(0.3) {
        // 預設搜尋範圍 [-100, 100]
        lowerBounds.resize(dim, -100.0);
        upperBounds.resize(dim, 100.0);
    }
    
    // 設定搜尋範圍
    void setBounds(const vector<double>& lower, const vector<double>& upper) {
        lowerBounds = lower;
        upperBounds = upper;
    }
};

// 適應度函數類型定義
using FitnessFunction = function<double(const vector<double>&)>;

// 常用的適應度函數（最小化問題）
namespace FitnessFunctions {
    double Sphere(const vector<double>& x);        // Sphere 函數
    double Ackley(const vector<double>& x);        // Ackley 函數
    double Rastrigin(const vector<double>& x);     // Rastrigin 函數
    double OneMax(const vector<double>& x);        // OneMax 問題（轉為最小化）
}

#endif 