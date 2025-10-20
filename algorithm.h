#include <vector>
#include <functional>
#include <random>
#include <ctime>
#include "structure.h"

#ifndef ALGORITHM_H
#define ALGORITHM_H

using namespace std;

// VisionNet 演算法主類別
class VisionNet
{
private:
    VisionNetParams params;         // 演算法參數
    HistoryTable historyTable;     // 歷史記憶表
    vector<vector<GridRegion>> grid; // L×L 網格結構
    vector<double> expectedValues;  // 期望值表 E
    GridPoint bestGridPoint;        // 網子最佳解 Xbest
    FitnessFunction fitnessFunc;   // 適應度函數
    mt19937 rng;                   // 隨機數生成器
    int evaluationCount;           // 函數評估次數
    int currentGeneration;         // 當前世代
    
    // D. Evaluation 相關
    vector<GridPoint> originalPoints; // 當前代更新前的原始點
    vector<GridPoint> updatedPoints;  // 當前代更新後的子網格點
    int currentUpdateRegion;          // 當前更新的網格區域ID
    
public:
    // 建構函數
    VisionNet(const VisionNetParams& parameters, FitnessFunction fitness);
    
    // 主要執行函數
    void RunVN();
    
    // 獲取結果
    GridPoint getBestSolution() const { return bestGridPoint; }
    double getBestFitness() const { return bestGridPoint.fitness; }
    int getEvaluationCount() const { return evaluationCount; }
    
    // 公用輔助函數
    pair<int, int> getGridCoord(int gridId) const { 
        int adjustedId = gridId - 1;  // 轉回0-based索引進行計算
        return {adjustedId / params.L, adjustedId % params.L}; 
    }
    
private:
    // 演算法核心函數
    void Initialization(int L, int Hsize);
    void ExpectedValue();
    void Net();
    vector<GridPoint> Evaluation(const vector<GridPoint>& x, const vector<GridPoint>& xe);
    
    // 期望值計算相關函數
    void calculateRawExpectedValues();
    void normalizeAndInvertExpectedValues();
    double calculateWeight() const;
    void updateVisitCounts(const vector<int>& selectedRegions);
    
    // 網子更新相關函數
    double calculatePbest() const;
    vector<int> selectTopRegions(double pbest);
    int selectRandomRegion(const vector<int>& topRegions);
    GridPoint selectXPbest(int regionId);
    pair<double, double> sampleFromHistory();
    double generateCr(double MCr);
    double generateF(double MF);
    GridPoint updateGridPoint(const GridPoint& xi, const GridPoint& xpbest, 
                             const GridPoint& xr1, const GridPoint& xr2, 
                             double Cr, double F, int jrand);
    void boundaryControl(GridPoint& point);
    GridPoint selectRandomGridPoint();  // 隨機選擇一個網格點
    
    // 輔助函數
    GridPoint createRandomGridPoint(int gridId);
    void evaluateGridPoint(GridPoint& point);
    void updateBestSolution(const GridPoint& candidate);
    int getGridId(int i, int j) const { return i * params.L + j + 1; }  // 從1開始編號
    void initializeGrid();
    void initializeExpectedValues();
    void initializeHistoryTable();
    bool shouldTerminate() const;
    void printProgress(int iteration) const;
    void printFinalResults() const;
    void updateGridWithEvaluatedPoints(const vector<GridPoint>& evaluatedPoints);
};

//原有的介面 (向後相容)
class algorithm 
{
public:
    void RunALG();  // 移除參數，使用預設值
    double get_best_fitness(int& best_idx) const;
    vector<double> get_best_position() const;
    
private:
    Solution bestSolution;
    double bestFitness;
    unique_ptr<VisionNet> visionNet;  // 使用 VisionNet 作為後端
};

#endif // ALGORITHM_H