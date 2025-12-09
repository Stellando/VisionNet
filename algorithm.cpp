
#include <ctime>
#include <cstdlib>
#include "algorithm.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <functional>

using namespace std;

// ==================== VisionNet 實作 ====================

VisionNet::VisionNet(const VisionNetParams& parameters, FitnessFunction fitness)
    : params(parameters), 
      fitnessFunc(fitness),
      evaluationCount(0),
      currentGeneration(0),
      history(parameters.Hsize, parameters.initialMCr, parameters.initialMF) {
    
    // === 修復問題 1：更強健的種子生成 ===
    std::random_device rd;
    
    // 獲取高精度時間
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    
    // 獲取記憶體位址 (確保即使同時間創建的不同物件也有不同種子)
    auto this_addr = reinterpret_cast<size_t>(this);
    
    // 混合多種熵源
    unsigned int seed1 = rd();
    unsigned int seed2 = static_cast<unsigned int>(nanos & 0xFFFFFFFF);
    unsigned int seed3 = static_cast<unsigned int>(this_addr & 0xFFFFFFFF);
    unsigned int seed4 = static_cast<unsigned int>((nanos >> 32) & 0xFFFFFFFF);
    
    // 使用 seed_seq 混合多個熵源
    std::seed_seq ss{seed1, seed2, seed3, seed4};
    
    rng.seed(ss);
    
    // 初始化最佳解容器
    bestSolution = Point(params.dimension, -1);
}

void VisionNet::RunVN() {
    cout << "=== Vision Net Algorithm Started ===" << endl;
    cout << "Net Size: " << params.L << "x" << params.L << " points" << endl;
    
    // A. 初始化
    Initialization();
    
    while (evaluationCount < params.maxEvaluations) {
        currentGeneration++;
        
        // B. 期望值計算
        ExpectedValue();
        
        // C. 網子更新
        vector<Point> trialPoints = Net();
        
        // D. 評估
        Evaluation(trialPoints);
        
        if (evaluationCount % 1000 == 0) { // 簡單進度顯示
             // cout << "Evals: " << evaluationCount << " Best: " << bestSolution.fitness << endl;
        }
    }
    
    printFinalResults();
}

void VisionNet::Initialization() {
    int totalPoints = params.L * params.L;
    points.resize(totalPoints, Point(params.dimension, 0));
    
    // 初始化所有點的位置與適應值
    uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < totalPoints; i++) {
        points[i].id = i;
        for (int d = 0; d < params.dimension; d++) {
            points[i].position[d] = params.lowerBounds[d] + 
                dis(rng) * (params.upperBounds[d] - params.lowerBounds[d]);
        }
        points[i].fitness = fitnessFunc(points[i].position);
        evaluationCount++;
        
        if (points[i].fitness < bestSolution.fitness) {
            bestSolution = points[i];
        }
    }
    
    // 建立網格結構 (L-1) * (L-1)
    int gridRows = params.L - 1;
    if (gridRows < 1) { cerr << "Error: L must be >= 2"; exit(1); }
    
    grids.resize(gridRows * gridRows);
    int gridCount = 0;
    
    // 依據圖 3.3 建立網格與點的關聯 (Grid 編號 1 ~ (L-1)^2)
    for (int i = 0; i < gridRows; i++) {     // Row index
        for (int j = 0; j < gridRows; j++) { // Col index
            GridRegion& g = grids[gridCount];
            g.id = gridCount;
            
            // 對應四個角點的索引 (在 points 陣列中)
            // Top-Left: i*L + j
            g.pointIndices[0] = i * params.L + j;
            // Top-Right: i*L + j + 1
            g.pointIndices[1] = i * params.L + (j + 1);
            // Bottom-Left: (i+1)*L + j
            g.pointIndices[2] = (i + 1) * params.L + j;
            // Bottom-Right: (i+1)*L + (j+1)
            g.pointIndices[3] = (i + 1) * params.L + (j + 1);
            
            gridCount++;
        }
    }
}

void VisionNet::ExpectedValue() {
    // 計算原始指標
    double EI_min = 1e9, EI_max = -1e9;
    double Eavg_min = 1e9, Eavg_max = -1e9;
    double Ebest_min = 1e9, Ebest_max = -1e9;
    
    for (auto& g : grids) {
        // EI: 造訪比例 (Ib / Ia)
        g.EI = (double)g.Ib / g.Ia;
        
        // Eavg: 平均目標值
        double sumFit = 0;
        double bestFitInGrid = 1e9;
        for(int k=0; k<4; k++) {
            double f = points[g.pointIndices[k]].fitness;
            sumFit += f;
            if(f < bestFitInGrid) bestFitInGrid = f;
        }
        g.Eavg = sumFit / 4.0; 
        g.Ebest = bestFitInGrid;
        
        // 紀錄 Max/Min 用於正規化
        if(g.EI < EI_min) EI_min = g.EI;
        if(g.EI > EI_max) EI_max = g.EI;
        
        if(g.Eavg < Eavg_min) Eavg_min = g.Eavg;
        if(g.Eavg > Eavg_max) Eavg_max = g.Eavg;
        
        if(g.Ebest < Ebest_min) Ebest_min = g.Ebest;
        if(g.Ebest > Ebest_max) Ebest_max = g.Ebest;
    }
    
    // 權重 w 計算 (隨時間遞減) Source 82
    double t_ratio = (double)evaluationCount / params.maxEvaluations; // 近似
    double w = (1.0 - 1.5) * t_ratio + 1.5;
    
    // 正規化與最終計算
    for (auto& g : grids) {
        double norm_EI = (EI_max == EI_min) ? 0 : (g.EI - EI_min) / (EI_max - EI_min);
        // 最小化問題：Eavg, Ebest 越小越好，正規化後希望它分數高，故反轉
        double norm_Eavg = (Eavg_max == Eavg_min) ? 0 : 1.0 - (g.Eavg - Eavg_min) / (Eavg_max - Eavg_min);
        double norm_Ebest = (Ebest_max == Ebest_min) ? 0 : 1.0 - (g.Ebest - Ebest_min) / (Ebest_max - Ebest_min);
        
        g.expectedValue = norm_EI + norm_Eavg + w * norm_Ebest;
    }
}

vector<Point> VisionNet::Net() {
    vector<Point> trialPoints = points; // 複製當前狀態
    
    // 3.1 計算 Pbest (Source 87)
    double t_ratio = (double)evaluationCount / params.maxEvaluations;
    double pbest_ratio = (0.2 - 0.4) * t_ratio + 0.4;
    
    // 3.2 選擇參考網格 (Reference Grid)
    // 依期望值排序
    vector<int> sortedGridIndices(grids.size());
    for(int i=0; i<grids.size(); i++) sortedGridIndices[i] = i;
    
    sort(sortedGridIndices.begin(), sortedGridIndices.end(), [&](int a, int b){
        return grids[a].expectedValue > grids[b].expectedValue; // 期望值越大越好
    });
    
    int topK = max(1, (int)(grids.size() * pbest_ratio));
    uniform_int_distribution<int> distTop(0, topK - 1);
    int selectedRank = distTop(rng);
    GridRegion& rPbestGrid = grids[sortedGridIndices[selectedRank]];
    
    // 3.3 更新造訪次數
    for(auto& g : grids) {
        if(g.id == rPbestGrid.id) {
            g.Ia++; g.Ib = 1;
        } else {
            g.Ib++; g.Ia = 1;
        }
    }
    
    // 3.4 選擇參考點 x_pbest (從選中網格的4點中隨機選1)
    uniform_int_distribution<int> dist4(0, 3);
    int pbestIdx = rPbestGrid.pointIndices[dist4(rng)];
    Point& x_pbest = points[pbestIdx];
    
    // 3.5 遍歷 **所有點** 進行更新 (Source 211: for i = 1 to |x|)
    for(int i = 0; i < points.size(); i++) {
        Point& xi = points[i];
        
        // 取得參數
        auto [MCr, MF] = history.getRandomEntry(rng);
        
        // === 修復問題 2：符合論文的機率分佈 [cite: 92, 95] ===
        
        // Cr 使用常態分佈 (Normal Distribution) Source 92
        normal_distribution<double> normCr(MCr, 0.1);
        double Cr = normCr(rng);
        Cr = min(1.0, max(0.0, Cr)); // 邊界限制 [0, 1] Source 93
        
        // F 使用柯西分佈 (Cauchy Distribution) Source 92
        cauchy_distribution<double> cauchyF(MF, 0.1);
        double F;
        
        // 根據 Source 95 處理 F 的邊界
        do {
            F = cauchyF(rng);
            if (F > 1.0) F = 1.0; // 若 > 1，設為 1
            // 若 < 0，do-while 會重新生成 (論文說 rand(MF, 0.1) 重新生成，這裡簡化為重抽)
        } while (F <= 0.0);
        
        // 紀錄參數供 Evaluation 使用
        trialPoints[i].Cr = Cr;
        trialPoints[i].F = F;
        
        // 選擇 r1, r2 (必須相異且不等於 i 和 pbest)
        int r1, r2;
        do { r1 = uniform_int_distribution<int>(0, points.size()-1)(rng); } while(r1 == i || r1 == pbestIdx);
        do { r2 = uniform_int_distribution<int>(0, points.size()-1)(rng); } while(r2 == i || r2 == pbestIdx || r2 == r1);
        
        // 進行向量更新 (DE/target-to-pbest/1)
        // 公式 Source 98: xi + F*(x_pbest - xi) + F*(x_r1 - x_r2)
        
        uniform_int_distribution<int> dimDist(0, params.dimension - 1);
        int jrand = dimDist(rng);
        uniform_real_distribution<double> rand01(0.0, 1.0);
        
        for(int j=0; j<params.dimension; j++) {
            if(rand01(rng) < Cr || j == jrand) {
                double val = xi.position[j] + 
                             F * (x_pbest.position[j] - xi.position[j]) + 
                             F * (points[r1].position[j] - points[r2].position[j]);
                
                // 邊界處理 (Source 101)
                if(val < params.lowerBounds[j] || val > params.upperBounds[j]) {
                    val = params.lowerBounds[j] + rand01(rng) * (params.upperBounds[j] - params.lowerBounds[j]);
                }
                trialPoints[i].position[j] = val;
            } else {
                trialPoints[i].position[j] = xi.position[j];
            }
        }
    }
    
    return trialPoints;
}

void VisionNet::Evaluation(vector<Point>& trialPoints) {
    vector<double> S_Cr, S_F, w_g;
    double total_diff = 0.0;
    
    for(int i=0; i<points.size(); i++) {
        // 評估新解
        double newFit = fitnessFunc(trialPoints[i].position);
        evaluationCount++;
        trialPoints[i].fitness = newFit;
        
        // 貪婪選擇
        if(newFit < points[i].fitness) { // 最小化問題
            double diff = points[i].fitness - newFit;
            points[i] = trialPoints[i]; // 取代舊解
            
            // 紀錄成功參數
            S_Cr.push_back(trialPoints[i].Cr);
            S_F.push_back(trialPoints[i].F);
            w_g.push_back(diff);
            total_diff += diff;
            
            // 更新全域最佳
            if(newFit < bestSolution.fitness) {
                bestSolution = points[i];
                // 記錄改進歷史
                improvementHistory.push_back({currentGeneration, bestSolution.fitness});
                // 實時報告找到更好解的進度 (小數點後兩位)
                std::cout << "世代 " << currentGeneration 
                          << ": 找到更好解，適應度 = " << std::scientific << std::setprecision(2) << bestSolution.fitness << std::endl;
            }
        }
    }
    
    // 更新歷史記憶表 (Lehmer Mean)
    if(!S_Cr.empty()) {
        double mean_Cr = 0, mean_F = 0;
        double den_Cr = 0, den_F = 0;
        
        // 計算 Lehmer Mean
        for(int k=0; k<S_Cr.size(); k++) {
            double weight = w_g[k] / total_diff;
            mean_Cr += weight * S_Cr[k]; 
            mean_F += weight * S_F[k] * S_F[k]; // Lehmer numerator
            den_F += weight * S_F[k];           // Lehmer denominator
        }
        if(den_F > 0) mean_F /= den_F;
        
        // 更新 History 中隨機一個位置 Source 104
        uniform_int_distribution<int> hDist(0, history.Hsize - 1);
        int k = hDist(rng);
        history.MCr_mem[k] = mean_Cr;
        history.MF_mem[k] = mean_F;
    }
}

void VisionNet::printFinalResults() const {
    cout << "\n=== Final Results ===" << endl;
    cout << "Total Generations: " << currentGeneration << endl;
    cout << "Total Evaluations: " << evaluationCount << endl;
    // 使用小數點後兩位顯示
    cout << "Best Fitness: " << std::scientific << std::setprecision(2) << bestSolution.fitness << endl;
    
    cout << "Best Position: [";
    for (int i = 0; i < bestSolution.position.size(); i++) {
        cout << fixed << setprecision(4) << bestSolution.position[i];
        if (i < bestSolution.position.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}