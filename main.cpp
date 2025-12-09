//g++ -g -std=c++17 main.cpp algorithm.cpp structure.cpp functions.cpp cec22_test_func.cpp -o main.exe 
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <string>
#include <map>
#ifdef _WIN32
#include <windows.h>
#endif
#include "algorithm.h"
#include "functions.h"

using namespace std;

// ==================== 測試設定區域 ====================
// CEC2022 測試函數設定
const int START_FUNC_ID = 4;              // 起始函數 ID (1-12)
const int END_FUNC_ID = 4;               // 結束函數 ID (1-12)

const int DIMENSION = 20;                 // 問題維度 (10 or 20 for CEC2022)
const int MAX_EVALUATIONS = DIMENSION * 20000;  // 最大評估次數
const int RUN_TIMES = 20;                 // 每個函數執行次數 (CEC 標準通常為 30)

// ==================== Vision Net 參數設定 ====================
const int VN_L = 11;                      // 網格邊長 (L*L 個點) - 依據 CEC'21 表格 (D=10 -> L=11)
const int VN_HSIZE = 2;                   // 歷史記憶表大小 - 依據 CEC'21 表格 (D=10 -> H=2)
const double VN_INIT_MCR = 0.1;           // 初始 MCr - 依據 CEC'21 表格
const double VN_INIT_MF = 0.3;            // 初始 MF - 依據 CEC'21 表格

// ==================== 輸出控制 ====================
const bool SAVE_TO_FILE = true;           // true=輸出TXT檔案, false=不輸出

// ==================== 中文顯示設定 ====================
void setupChineseDisplay() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    setlocale(LC_ALL, ".UTF8");
#endif
}

// ==================== 主程式 ====================
int main() {
    setupChineseDisplay();
    
    cout << "=======================================================" << endl;
    cout << "  Vision Net Algorithm - CEC2022 Benchmark" << endl;
    cout << "=======================================================" << endl;
    cout << "維度: " << DIMENSION << " | 評估次數: " << MAX_EVALUATIONS << endl;
    cout << "測試範圍: F" << START_FUNC_ID << " - F" << END_FUNC_ID << endl;
    cout << "每函數執行: " << RUN_TIMES << " 次 | 檔案輸出: " << (SAVE_TO_FILE ? "ON" : "OFF") << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "VN 參數: L=" << VN_L << " (" << VN_L*VN_L << " points) | Hsize=" << VN_HSIZE << endl;
    cout << "         Init MCr=" << VN_INIT_MCR << " | Init MF=" << VN_INIT_MF << endl;
    cout << "=======================================================" << endl << endl;
    
    // 對每個測試函數執行
    for(int funcId = START_FUNC_ID; funcId <= END_FUNC_ID; funcId++) {
        // 獲取函數資訊
        FunctionInfo funcInfo = CEC22Functions::getFunctionInfo(funcId, DIMENSION);
        
        cout << "\n┌─────────────────────────────────────────────────────┐" << endl;
        cout << "│ " << left << setw(52) << funcInfo.name << "│" << endl;
        cout << "│ " << left << setw(52) << funcInfo.description << "│" << endl;
        cout << "└─────────────────────────────────────────────────────┘" << endl;
        
        // 儲存結果
        vector<double> results;
        vector<double> times;
        
        // 儲存所有 Run 的改進歷史 (用於計算平均)
        map<int, vector<double>> allImprovements; // generation -> list of fitness values from all runs
        
        // 執行測試
        for(int run = 1; run <= RUN_TIMES; run++) {
            cout << "  Run " << setw(2) << run << "/" << RUN_TIMES << "..." << flush;
            
            auto start = chrono::high_resolution_clock::now();
            
            // 設定參數
            VisionNetParams params(VN_L, DIMENSION);
            params.maxEvaluations = MAX_EVALUATIONS;
            params.Hsize = VN_HSIZE;
            params.initialMCr = VN_INIT_MCR;
            params.initialMF = VN_INIT_MF;
            
            // 設定搜尋範圍
            vector<double> lower(DIMENSION, funcInfo.searchRange[0]);
            vector<double> upper(DIMENSION, funcInfo.searchRange[1]);
            params.setBounds(lower, upper);
            
            // 建立並執行 Vision Net
            VisionNet vn(params, funcInfo.func);
            vn.RunVN();
            
            auto end = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(end - start).count();
            
            // 獲取結果 (計算與全域最佳值的誤差)
            double fitness = vn.getBestFitness();
            double error = abs(fitness - funcInfo.globalOptimum);
            
            // 移除人為截斷，保留原始精度
            // if (error < 1e-15) error = 0.0;
            
            results.push_back(error);
            times.push_back(elapsed);
            
            // 收集改進歷史
            auto history = vn.getImprovementHistory();
            for(const auto& [gen, fitness] : history) {
                // 儲存誤差值 (Error = |Fitness - Bias|)
                double errorVal = abs(fitness - funcInfo.globalOptimum);
                allImprovements[gen].push_back(errorVal);
            }
            
            cout << " ✓ Error: " << scientific << setprecision(4) << error 
                 << " (" << fixed << setprecision(2) << elapsed << "s)" << endl;
        }
        
        // 統計分析
        double mean = accumulate(results.begin(), results.end(), 0.0) / RUN_TIMES;
        double variance = 0.0;
        for(double val : results) variance += (val - mean) * (val - mean);
        double stdDev = sqrt(variance / RUN_TIMES);
        double best = *min_element(results.begin(), results.end());
        double worst = *max_element(results.begin(), results.end());
        double avgTime = accumulate(times.begin(), times.end(), 0.0) / RUN_TIMES;
        
        // 顯示統計結果
        cout << "\n  " << string(49, '-') << endl;
        cout << "  統計結果 (Error):" << endl;
        cout << "  " << string(49, '-') << endl;
        cout << scientific << setprecision(6);
        cout << "  Best:  " << best << endl;
        cout << "  Worst: " << worst << endl;
        cout << "  Mean:  " << mean << endl;
        cout << "  Std:   " << stdDev << endl;
        cout << fixed << setprecision(3);
        cout << "  Time:  " << avgTime << " s" << endl;
        cout << "  " << string(49, '-') << endl;
        
        // 檔案輸出
        if(SAVE_TO_FILE) {
            // 產生檔案名稱 (將空格替換為底線)
            string safeName = funcInfo.name;
            replace(safeName.begin(), safeName.end(), ' ', '_');
            
            string filename = "VN_" + safeName + "_D" + to_string(DIMENSION) + ".txt";
            ofstream file(filename);
            
            if(file.is_open()) {
                file << "Vision Net Algorithm Test Results" << endl;
                file << "Function: " << funcInfo.name << " | Dimension: " << DIMENSION << endl;
                file << "Evaluations: " << MAX_EVALUATIONS << " | Runs: " << RUN_TIMES << endl;
                file << "Parameters: L=" << VN_L << ", Hsize=" << VN_HSIZE 
                     << ", MCr=" << VN_INIT_MCR << ", MF=" << VN_INIT_MF << endl;
                file << "========================================" << endl << endl;
                
                file << scientific << setprecision(6);
                file << "Best Error:  " << best << endl;
                file << "Worst Error: " << worst << endl;
                file << "Mean Error:  " << mean << endl;
                file << "Std Dev:     " << stdDev << endl;
                file << fixed << setprecision(3);
                file << "Avg Time:    " << avgTime << " s" << endl << endl;
                
                file << "All Run Errors:" << endl;
                file << scientific << setprecision(6);
                for(int i = 0; i < RUN_TIMES; i++) {
                    file << "Run " << setw(2) << (i+1) << ": " << results[i] << endl;
                }
                
                file.close();
                cout << "\n  ✓ 結果已儲存: " << filename << endl;
            }
            
            // 輸出平均改進歷史
            string historyFilename = "VN_" + safeName + "_D" + to_string(DIMENSION) + "_history.txt";
            ofstream historyFile(historyFilename);
            
            if(historyFile.is_open()) {
                // historyFile << "# Average Improvement History (Generation Fitness)" << endl;
                // historyFile << "# Function: " << funcInfo.name << " | Dimension: " << DIMENSION << endl;
                // historyFile << "# Averaged over " << RUN_TIMES << " runs" << endl;
                // historyFile << "# Format: Generation AvgFitness" << endl;
                // historyFile << "========================================" << endl;
                
                // 按世代排序並輸出平均值
                for(const auto& [gen, fitnessList] : allImprovements) {
                    double avgFitness = accumulate(fitnessList.begin(), fitnessList.end(), 0.0) / fitnessList.size();
                    historyFile << gen << " " << scientific << setprecision(2) << avgFitness << endl;
                }
                
                historyFile.close();
                cout << "  ✓ 改進歷史已儲存: " << historyFilename << endl;
            }
        }
    }
    
    cout << "\n=======================================================" << endl;
    cout << "  所有測試完成！" << endl;
    cout << "=======================================================" << endl;
    system("pause");
    return 0;
}

