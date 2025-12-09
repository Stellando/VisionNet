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
    FitnessFunction fitnessFunc;    // 適應度函數
    mt19937 rng;                    // 隨機數生成器
    int evaluationCount;            // 函數評估次數
    int currentGeneration;          // 當前世代

    // 資料結構修正
    vector<Point> points;           // 總共 L*L 個點
    vector<GridRegion> grids;       // 總共 (L-1)*(L-1) 個網格
    HistoryTable history;           // 歷史記憶表
    
    Point bestSolution;             // 最佳解
    
    // 追蹤改進歷史 (世代, 適應度)
    vector<pair<int, double>> improvementHistory;

public:
    // 建構函數
    VisionNet(const VisionNetParams& parameters, FitnessFunction fitness);
    
    // 主要執行函數
    void RunVN();
    
    // 獲取結果
    Point getBestSolution() const { return bestSolution; }
    double getBestFitness() const { return bestSolution.fitness; }
    int getEvaluationCount() const { return evaluationCount; }
    vector<pair<int, double>> getImprovementHistory() const { return improvementHistory; }
    
private:
    // 演算法核心函數
    void Initialization();
    void ExpectedValue();
    vector<Point> Net();
    void Evaluation(vector<Point>& trialPoints);
    
    void printFinalResults() const;
};

#endif // ALGORITHM_H