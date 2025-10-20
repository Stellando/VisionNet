
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

using namespace std;

// ==================== VisionNet 實作 ====================

VisionNet::VisionNet(const VisionNetParams& parameters, FitnessFunction fitness)
    : params(parameters), 
      historyTable(parameters.Hsize, parameters.dimension),
      fitnessFunc(fitness),
      rng(time(nullptr)),
      evaluationCount(0),
      currentGeneration(0) {
    
    // 初始化網格結構
    grid.resize(params.L, vector<GridRegion>(params.L));
    expectedValues.resize(params.L * params.L, 0.0);
    bestGridPoint = GridPoint(params.dimension, -1);
    bestGridPoint.fitness = numeric_limits<double>::infinity();  // 最小化問題：初始為無窮大
}

void VisionNet::RunVN() {
    cout << "=== Vision Net Algorithm Started ===" << endl;
    cout << "Grid Size: " << params.L << "x" << params.L << " (" << params.L * params.L << " regions)" << endl;
    cout << "Dimension: " << params.dimension << endl;
    cout << "Max Evaluations: " << params.maxEvaluations << endl;
    
    // A. 初始化
    Initialization(params.L, params.Hsize);
    
    cout << "Initialization completed. Best initial fitness: " << bestGridPoint.fitness << endl;
    
    // 主要迭代循環
    while (evaluationCount < params.maxEvaluations) {
        currentGeneration++;
        //cout << "\n--- Generation " << currentGeneration << " ---" << endl;
        // B. 期望值計算
        ExpectedValue();
        
        // C. 網子更新
        Net();
        
        // D. 評估
        vector<GridPoint> evaluatedPoints = Evaluation(originalPoints, updatedPoints);
        
        // 將評估後的結果更新回網格
        updateGridWithEvaluatedPoints(evaluatedPoints);
        /*
        // 每100回合輸出結果
        if (currentGeneration % 100 == 0) {
            printProgress(currentGeneration);
        }
        */
        // 檢查終止條件
        if (shouldTerminate()) {
            cout << "Early termination at generation " << currentGeneration << endl;
            break;
        }
    }
    
    printFinalResults();
}

void VisionNet::Initialization(int L, int Hsize) {
    cout << "Starting initialization..." << endl;
    
    // 1. 初始化網格結構
    initializeGrid();
    
    // 2. 初始化期望值表
    initializeExpectedValues();
    
    // 3. 初始化歷史記憶表
    initializeHistoryTable();
    
    // 4. 對所有網格點進行評估，找出最佳解
    cout << "Evaluating all grid points..." << endl;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (auto& point : grid[i][j].points) {
                evaluateGridPoint(point);
                updateBestSolution(point);
            }
        }
    }
    
    cout << "Initialization completed. Total evaluations: " << evaluationCount << endl;
    
    // 保存初始網格點作為第一代
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            grid[i][j].previousPoints = grid[i][j].points;  // 保存當前點作為前一代
        }
    }
}

void VisionNet::initializeGrid() {
    uniform_real_distribution<double> dis(0.0, 1.0);
    
    cout << "Creating grid with region numbering from 1 to " << params.L * params.L << " (left-to-right, top-to-bottom)" << endl;
    
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            int gridId = getGridId(i, j);  // 從1開始編號
            grid[i][j].regionId = gridId;
            grid[i][j].gridCoord = {i, j};
            grid[i][j].expectedValue = 0.0;
            
            // 每個網格創建4個網格點（方形的四個頂點）
            // 在網格內創建四個角點位置
            for (int corner = 0; corner < 4; corner++) {
                GridPoint point(params.dimension, gridId);
                
                // 對每個維度進行隨機初始化
                // Xi,j = min(xj) + rand(0,1) * (max(xj) - min(xj))
                // 其中 i 是個體索引，j 是維度索引
                for (int d = 0; d < params.dimension; d++) {
                    double randVal = dis(rng);
                    point.position[d] = params.lowerBounds[d] + 
                                       randVal * (params.upperBounds[d] - params.lowerBounds[d]);
                }
                
                grid[i][j].points.push_back(point);
            }
            
            // 輸出前幾個區域的編號來驗證
            if (i < 2 && j < 3) {
                cout << "Grid[" << i << "][" << j << "] -> Region ID: " << gridId << endl;
            }
        }
    }
    
    cout << "Grid structure initialized: " << params.L * params.L << " regions (ID 1-" 
         << params.L * params.L << ") with " << params.L * params.L * 4 << " grid points." << endl;
}

void VisionNet::initializeExpectedValues() {
    // 期望值表 E：每個網格區域對應一個期望值，初始為 0
    fill(expectedValues.begin(), expectedValues.end(), 0.0);
    cout << "Expected values table initialized (all zeros)." << endl;
}

void VisionNet::initializeHistoryTable() {
    // 歷史記憶表 H：長度 |H| = Hsize × d，存儲 MCr 和 MF 值
    historyTable.clear();
    
    int totalEntries = params.Hsize * params.dimension;
    cout << "Initializing history table with |H| = " << params.Hsize 
         << " × " << params.dimension << " = " << totalEntries << " entries" << endl;
    
    // 為每個維度創建 Hsize 個參數條目
    for (int d = 0; d < params.dimension; d++) {
        for (int h = 0; h < params.Hsize; h++) {
            HistoryParameter param(params.initialMCr, params.initialMF, d, 0);
            historyTable.addParameter(param);
        }
    }
    
    cout << "History table initialized: " << historyTable.size() << " entries with MCr=" 
         << params.initialMCr << ", MF=" << params.initialMF << endl;
}

GridPoint VisionNet::createRandomGridPoint(int gridId) {
    uniform_real_distribution<double> dis(0.0, 1.0);
    GridPoint point(params.dimension, gridId);
    
    for (int d = 0; d < params.dimension; d++) {
        double randVal = dis(rng);
        point.position[d] = params.lowerBounds[d] + 
                           randVal * (params.upperBounds[d] - params.lowerBounds[d]);
    }
    
    return point;
}

void VisionNet::evaluateGridPoint(GridPoint& point) {
    point.fitness = fitnessFunc(point.position);
    evaluationCount++;
}

void VisionNet::updateBestSolution(const GridPoint& candidate) {
    // 注意：因為是最小化問題，所以比較條件是 <
    if (candidate.fitness < bestGridPoint.fitness) {
        bestGridPoint = candidate;
        // 實時報告找到更好解的進度
        std::cout << "世代 " << currentGeneration 
                  << ": 找到更好解，適應度 = " << candidate.fitness << std::endl;
    }
}

// ==================== 期望值計算實作 ====================

void VisionNet::ExpectedValue() {
    //cout << "Starting expected value calculation for generation " << currentGeneration << "..." << endl;
    
    // 1. 計算原始期望值（EI, Eavg, Ebest）
    calculateRawExpectedValues();
    
    // 2. 進行最小值-最大值正規化並反轉
    normalizeAndInvertExpectedValues();
    
    // 3. 計算權重 w
    double weight = calculateWeight();
    
    // 4. 計算最終期望值 Ei = EI,i + Eavg,i + w × Ebest,i
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            GridRegion& region = grid[i][j];
            region.expectedValue = region.EI_normalized + region.Eavg_normalized + weight * region.Ebest_normalized;
            
            // 更新期望值表
            int regionIndex = region.regionId - 1;  // 轉回0-based索引
            expectedValues[regionIndex] = region.expectedValue;
        }
    }
    
    //cout << "Expected value calculation completed. Weight w = " << fixed << setprecision(4) << weight << endl;
}

void VisionNet::calculateRawExpectedValues() {
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            GridRegion& region = grid[i][j];
            
            // 1. 計算 EI,i = Ib / Ia（造訪比例）
            region.EI = static_cast<double>(region.Ib) / static_cast<double>(region.Ia);
            
            // 2. 計算 Eavg,i（平均增進值）
            if (currentGeneration == 1) {
                // 第一次迭代時，沒有前一代，設為0
                region.Eavg = 0.0;
            } else {
                double totalImprovement = 0.0;
                for (int n = 0; n < 4; n++) {
                    double currentFitness = region.points[n].fitness;
                    double previousFitness = region.previousPoints[n].fitness;
                    totalImprovement += (currentFitness - previousFitness);
                }
                region.Eavg = totalImprovement / 4.0;
            }
            
            // 3. 計算 Ebest,i（最佳目標值）
            double bestFitness = numeric_limits<double>::infinity();
            for (const auto& point : region.points) {
                if (point.fitness < bestFitness) {  // 最小化問題
                    bestFitness = point.fitness;
                }
            }
            region.Ebest = bestFitness;
        }
    }
}

void VisionNet::normalizeAndInvertExpectedValues() {
    // 找出所有網格中 EI, Eavg, Ebest 的最大值和最小值
    double EI_min = numeric_limits<double>::infinity();
    double EI_max = -numeric_limits<double>::infinity();
    double Eavg_min = numeric_limits<double>::infinity();
    double Eavg_max = -numeric_limits<double>::infinity();
    double Ebest_min = numeric_limits<double>::infinity();
    double Ebest_max = -numeric_limits<double>::infinity();
    
    // 第一次遍歷：找出最大值和最小值
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            const GridRegion& region = grid[i][j];
            
            EI_min = min(EI_min, region.EI);
            EI_max = max(EI_max, region.EI);
            Eavg_min = min(Eavg_min, region.Eavg);
            Eavg_max = max(Eavg_max, region.Eavg);
            Ebest_min = min(Ebest_min, region.Ebest);
            Ebest_max = max(Ebest_max, region.Ebest);
        }
    }
    
    // 第二次遍歷：進行正規化和反轉
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            GridRegion& region = grid[i][j];
            
            // Min-Max 正規化：E限縮 = (E - Emin) / (Emax - Emin)
            double EI_normalized = (EI_max == EI_min) ? 0.0 : (region.EI - EI_min) / (EI_max - EI_min);
            double Eavg_normalized = (Eavg_max == Eavg_min) ? 0.0 : (region.Eavg - Eavg_min) / (Eavg_max - Eavg_min);
            double Ebest_normalized = (Ebest_max == Ebest_min) ? 0.0 : (region.Ebest - Ebest_min) / (Ebest_max - Ebest_min);
            
            // 反轉（因為最小化問題）：E反轉 = 1 - E限縮
            region.EI_normalized = 1.0 - EI_normalized;
            region.Eavg_normalized = 1.0 - Eavg_normalized;
            region.Ebest_normalized = 1.0 - Ebest_normalized;
        }
    }
}

double VisionNet::calculateWeight() const {
    // w = (1.0 - 1.5) / tmax * t + 1.5
    // w = -0.5 / tmax * t + 1.5
    double t = static_cast<double>(currentGeneration);
    double tmax = static_cast<double>(params.maxEvaluations / (params.L * params.L * 4)); // 估算最大世代數
    
    double weight = (-0.5 / tmax) * t + 1.5;
    
    // 確保權重在合理範圍內
    weight = max(1.0, min(1.5, weight));
    
    return weight;
}

void VisionNet::updateVisitCounts(const vector<int>& selectedRegions) {
    // 重設所有網格的訪問計數
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            GridRegion& region = grid[i][j];
            
            // 檢查這個區域是否被選取進行更新
            bool isSelected = false;
            for (int regionId : selectedRegions) {
                if (region.regionId == regionId) {
                    isSelected = true;
                    break;
                }
            }
            
            if (isSelected) {
                // 被選取：Ia增加，Ib重設為1
                region.Ia++;
                region.Ib = 1;
            } else {
                // 未被選取：Ib增加，Ia重設為1
                region.Ib++;
                region.Ia = 1;
            }
        }
    }
}

// ==================== 網子更新實作 ====================

void VisionNet::Net() {
    //cout << "Starting net update for generation " << currentGeneration << "..." << endl;
    
    // 清空上一代的點數據
    originalPoints.clear();
    updatedPoints.clear();
    
    // 1. 計算 Pbest 比例
    double pbest = calculatePbest();
    
    // 2. 選擇前 Pbest 的網格
    vector<int> topRegions = selectTopRegions(pbest);
    
    // 3. 從前 Pbest 中隨機選擇一個網格 rPbest
    int rPbest = selectRandomRegion(topRegions);
    currentUpdateRegion = rPbest;  // 記錄當前更新的區域
    
    // 4. 從 rPbest 網格的四個頂點中隨機選擇一點作為 XPbest
    GridPoint xPbest = selectXPbest(rPbest);
    
    // 5. 更新選中網格的 Ia 和 Ib
    vector<int> selectedRegions = {rPbest};
    updateVisitCounts(selectedRegions);
    
    // 6. 只更新選中的 rPbest 網格中的4個點
    // 找到對應的網格並進行更新
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            GridRegion& region = grid[i][j];
            
            // 保存當前點作為前一代（用於下次期望值計算）
            region.previousPoints = region.points;
            
            // 只更新選中的網格
            if (region.regionId == rPbest) {
                //cout << "Updating selected grid region " << rPbest << " with 4 points..." << endl;
                
                // 更新該網格的每個點（4個頂點）
                for (int pointIdx = 0; pointIdx < 4; pointIdx++) {
                    GridPoint& xi = region.points[pointIdx];
                    
                    // 保存原始點（更新前）
                    originalPoints.push_back(xi);
                    
                    // 從歷史記憶表中隨機選擇 MCr 和 MF
                    auto [MCr, MF] = sampleFromHistory();
                    
                    // 生成 Cr 和 F
                    double Cr = generateCr(MCr);
                    double F = generateF(MF);
                    
                    // 隨機選擇 r1 和 r2（確保與 xi 和 xPbest 不同）
                    GridPoint xr1, xr2;
                    do {
                        xr1 = selectRandomGridPoint();
                    } while (xr1.gridId == xi.gridId || xr1.gridId == xPbest.gridId);
                    
                    do {
                        xr2 = selectRandomGridPoint();
                    } while (xr2.gridId == xi.gridId || xr2.gridId == xPbest.gridId || xr2.gridId == xr1.gridId);
                    
                    // 隨機產生 jrand
                    uniform_int_distribution<int> jrandDist(0, params.dimension - 1);
                    int jrand = jrandDist(rng);
                    
                    // 使用 DE 算法更新網格點
                    GridPoint newXi = updateGridPoint(xi, xPbest, xr1, xr2, Cr, F, jrand);
                    
                    // 邊界控制
                    boundaryControl(newXi);
                    
                    // 保持網格ID
                    newXi.gridId = xi.gridId;
                    
                    // 保存更新後的點（但先不替換原來的點）
                    updatedPoints.push_back(newXi);
                }
                break;  // 找到並更新了選中的網格，跳出循環
            }
        }
    }
    
    // 7. 其他未被選中的網格不進行向量更新，但Ia/Ib會在updateVisitCounts中處理
    
    //cout << "Net update completed. Pbest = " << fixed << setprecision(4) << pbest << ", selected region: " << rPbest << endl;
}

double VisionNet::calculatePbest() const {
    // Pbest = (0.2 - 0.4) / tmax * t + 0.4
    double t = static_cast<double>(currentGeneration);
    double tmax = static_cast<double>(params.maxEvaluations / (params.L * params.L * 4));
    
    double pbest = (0.2 - 0.4) / tmax * t + 0.4;
    // pbest = -0.2 / tmax * t + 0.4
    
    // 確保 Pbest 在合理範圍內
    pbest = max(0.2, min(0.4, pbest));
    
    return pbest;
}

vector<int> VisionNet::selectTopRegions(double pbest) {
    // 收集所有網格及其期望值
    vector<pair<double, int>> regionExpectedValues;
    
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            const GridRegion& region = grid[i][j];
            regionExpectedValues.emplace_back(region.expectedValue, region.regionId);
        }
    }
    
    // 按期望值降序排序
    sort(regionExpectedValues.begin(), regionExpectedValues.end(), 
         [](const pair<double, int>& a, const pair<double, int>& b) {
             return a.first > b.first;
         });
    
    // 選擇前 Pbest 比例的網格
    int topCount = max(1, static_cast<int>(regionExpectedValues.size() * pbest));
    
    vector<int> topRegions;
    for (int i = 0; i < topCount; i++) {
        topRegions.push_back(regionExpectedValues[i].second);
    }
    
    return topRegions;
}

int VisionNet::selectRandomRegion(const vector<int>& topRegions) {
    uniform_int_distribution<int> dist(0, topRegions.size() - 1);
    int index = dist(rng);
    return topRegions[index];
}

GridPoint VisionNet::selectXPbest(int regionId) {
    // 找到對應的網格
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            if (grid[i][j].regionId == regionId) {
                // 從該網格的4個點中隨機選擇一個
                uniform_int_distribution<int> pointDist(0, 3);
                int pointIndex = pointDist(rng);
                return grid[i][j].points[pointIndex];
            }
        }
    }
    
    // 備用：如果沒找到，返回一個隨機點
    return createRandomGridPoint(regionId);
}

pair<double, double> VisionNet::sampleFromHistory() {
    // 從歷史記憶表中隨機選擇一組 MCr 和 MF
    // r 為 1 至 H 的隨機值，其中 H = |H| = Hsize × d = 200
    int H = params.Hsize * params.dimension;  // H = 20 × 10 = 200
    
    if (historyTable.size() == 0) {
        return {params.initialMCr, params.initialMF};
    }
    
    uniform_int_distribution<int> dist(0, min(H, historyTable.size()) - 1);
    int r = dist(rng);  // r為隨機索引值
    
    auto recentParams = historyTable.getRecentParameters(historyTable.size());
    if (r < recentParams.size()) {
        return {recentParams[r].MCr, recentParams[r].MF};
    }
    
    // 備用：返回預設值
    return {params.initialMCr, params.initialMF};
}

double VisionNet::generateCr(double MCr) {
    // Cr,i = rand(MCr,r, 0.1, n) 以標準差為 0.1 的常態分布
    normal_distribution<double> normalDist(MCr, 0.1);
    double Cr = normalDist(rng);
    
    // 邊界控制：限制在 [0, 1] 之間
    Cr = max(0.0, min(1.0, Cr));
    
    return Cr;
}

double VisionNet::generateF(double MF) {
    // Fi = rand(MF,r, 0.1, c) 以標準差為 0.1 的常態分布
    normal_distribution<double> normalDist(MF, 0.1);
    double F;
    
    do {
        F = normalDist(rng);
    } while (F < 0.0);  // 若小於0則重新生成
    
    return F;
}

GridPoint VisionNet::selectRandomGridPoint() {
    uniform_int_distribution<int> iDist(0, params.L - 1);
    uniform_int_distribution<int> jDist(0, params.L - 1);
    uniform_int_distribution<int> pointDist(0, 3);
    
    int i = iDist(rng);
    int j = jDist(rng);
    int pointIdx = pointDist(rng);
    
    return grid[i][j].points[pointIdx];
}

GridPoint VisionNet::updateGridPoint(const GridPoint& xi, const GridPoint& xpbest, 
                                     const GridPoint& xr1, const GridPoint& xr2, 
                                     double Cr, double F, int jrand) {
    GridPoint newXi = xi;  // 複製原始點
    
    // 保存使用的參數值
    newXi.Cr = Cr;
    newXi.F = F;
    
    uniform_real_distribution<double> uniformDist(0.0, 1.0);
    
    // 對每個維度進行更新
    for (int j = 0; j < params.dimension; j++) {
        double rand01 = uniformDist(rng);
        
        // 若隨機值小於交配率 Cr 或更新的維度 j = jrand
        if (rand01 < Cr || j == jrand) {
            // 使用經典DE算法：NEWxi,j = xi,j + Fi·(xpbest,j − xi,j) + Fi·(xr1,j − xr2,j)
            newXi.position[j] = xi.position[j] + 
                               F * (xpbest.position[j] - xi.position[j]) + 
                               F * (xr1.position[j] - xr2.position[j]);
        }
        // 否則保持原值（已經在 newXi = xi 中設定）
    }
    
    return newXi;
}

void VisionNet::boundaryControl(GridPoint& point) {
    uniform_real_distribution<double> uniformDist(0.0, 1.0);
    
    // 檢查每個維度是否超出解空間範圍
    for (int j = 0; j < params.dimension; j++) {
        if (point.position[j] < params.lowerBounds[j] || point.position[j] > params.upperBounds[j]) {
            // 若超出則進行隨機賦值
            // NEWxi,j = min(xj) + rand(0,1) × (max(xj) − min(xj))
            double randVal = uniformDist(rng);
            point.position[j] = params.lowerBounds[j] + 
                               randVal * (params.upperBounds[j] - params.lowerBounds[j]);
        }
    }
}

bool VisionNet::shouldTerminate() const {
    return evaluationCount >= params.maxEvaluations;
}

void VisionNet::printProgress(int iteration) const {
    cout << "Generation " << iteration 
         << " | Evaluations: " << evaluationCount 
         << " | Best Fitness: " << bestGridPoint.fitness << endl;
}

void VisionNet::printFinalResults() const {
    cout << "\n=== Final Results ===" << endl;
    cout << "Total Generations: " << currentGeneration << endl;
    cout << "Total Evaluations: " << evaluationCount << endl;
    cout << "Best Fitness: " << bestGridPoint.fitness << endl;
    cout << "Best Grid Region ID: " << bestGridPoint.gridId << " (numbered from 1)" << endl;
    
    auto coord = getGridCoord(bestGridPoint.gridId);
    cout << "Best Grid Coordinate: (" << coord.first << ", " << coord.second << ") in grid[i][j] format" << endl;
    
    cout << "Best Position: [";
    for (int i = 0; i < bestGridPoint.position.size(); i++) {
        cout << fixed << setprecision(4) << bestGridPoint.position[i];
        if (i < bestGridPoint.position.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}

// D. Evaluation - 評估步驟-------------------------------------------------------------------


vector<GridPoint> VisionNet::Evaluation(const vector<GridPoint>& x, const vector<GridPoint>& xe) {
    //cout << "Starting evaluation for " << xe.size() << " updated points..." << endl;
    
    // 建立成功表
    vector<double> SCr;  // 成功交配率表
    vector<double> SF;   // 成功比例係數表
    vector<double> weights; // 權重表
    vector<GridPoint> result = x; // 複製原始點作為結果
    
    double totalImprovement = 0.0;  // 總改善量（用於計算權重）
    
    // 評估所有子網格點並記錄成功案例
    for (size_t i = 0; i < xe.size(); i++) {
        // 評估子網格點
        GridPoint evaluatedPoint = xe[i];
        evaluateGridPoint(evaluatedPoint);
        
        // 檢查是否優於原網格點（最小化問題：新適應度 < 原適應度）
        double oldFitness = x[i].fitness;
        double newFitness = evaluatedPoint.fitness;
        
        if (newFitness < oldFitness) {
            // 成功更新：記錄到成功表
            double improvement = abs(oldFitness - newFitness);
            totalImprovement += improvement;
            
            SCr.push_back(evaluatedPoint.Cr);
            SF.push_back(evaluatedPoint.F);
            weights.push_back(improvement);  // 暫存改善量，後續計算權重
            
            // 更新結果點
            result[i] = evaluatedPoint;
            
            // 更新全域最佳解
            updateBestSolution(evaluatedPoint);
            
            //cout << "Point " << i << " improved: " << oldFitness << " -> " << newFitness << " (Cr=" << evaluatedPoint.Cr << ", F=" << evaluatedPoint.F << ")" << endl;
        }
        // 如果沒有改善，保持原來的點
    }
    
    //cout << "Evaluation completed. Success count: " << SCr.size() << "/" << xe.size() << endl;
    
    // 如果有成功更新，則更新歷史記憶表
    if (!SCr.empty() && totalImprovement > 0) {
        //cout << "Updating history table with " << SCr.size() << " successful updates..." << endl;
        
        // 計算正規化權重 wn = improvement_n / total_improvement
        for (double& w : weights) {
            w = w / totalImprovement;
        }
        
        // 計算萊默加權平均更新 MCr
        double numeratorCr = 0.0, denominatorCr = 0.0;
        for (size_t i = 0; i < SCr.size(); i++) {
            numeratorCr += weights[i] * SCr[i] * SCr[i];   // wn * SCr^2
            denominatorCr += weights[i] * SCr[i];          // wn * SCr
        }
        
        double newMCr = (denominatorCr > 0) ? (numeratorCr / denominatorCr) : 0.0;
        
        // 計算萊默加權平均更新 MF
        double numeratorF = 0.0, denominatorF = 0.0;
        for (size_t i = 0; i < SF.size(); i++) {
            numeratorF += weights[i] * SF[i] * SF[i];      // wn * SF^2
            denominatorF += weights[i] * SF[i];            // wn * SF
        }
        
        double newMF = (denominatorF > 0) ? (numeratorF / denominatorF) : 0.0;
        
        // 隨機選擇歷史表位置k進行更新
        int k = uniform_int_distribution<int>(0, historyTable.getSize() - 1)(rng);
        
        // 只在分母不為0時更新歷史表
        if (denominatorCr > 0) {
            historyTable.updateMCr(k, newMCr);
            //cout << "Updated MCr[" << k << "] = " << newMCr << endl;
        }
        
        if (denominatorF > 0) {
            historyTable.updateMF(k, newMF);
            //cout << "Updated MF[" << k << "] = " << newMF << endl;
        }
    } else {
        //cout << "No successful updates - history table unchanged" << endl;
    }
    
    return result;
}

// 將評估後的點更新回網格
void VisionNet::updateGridWithEvaluatedPoints(const vector<GridPoint>& evaluatedPoints) {
    //cout << "Updating grid with " << evaluatedPoints.size() << " evaluated points..." << endl;
    
    // 找到當前更新的網格區域
    for (int i = 0; i < params.L; i++) {
        for (int j = 0; j < params.L; j++) {
            GridRegion& region = grid[i][j];
            
            if (region.regionId == currentUpdateRegion) {
                // 更新該區域的點
                for (int pointIdx = 0; pointIdx < evaluatedPoints.size() && pointIdx < 4; pointIdx++) {
                    region.points[pointIdx] = evaluatedPoints[pointIdx];
                }
                
                //cout << "Grid region " << currentUpdateRegion << " updated with evaluated points" << endl;
                return;
            }
        }
    }
}