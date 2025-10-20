#include <iostream>
#include <iomanip>
#include <chrono>
#include <locale>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif
#include "algorithm.h"
#include "structure.h"
#include "functions.h"
using namespace std;
// ==================== 測試設定區域 ====================
// 只需要修改這裡就能切換不同的測試函數！

const string FUNCTION_NAME = "Ackley";    // 🔧 在這裡改變測試函數！
const int DIMENSION = 30;                   // 🔧 在這裡改變問題維度！

// 可選函數列表：
// "Sphere", "Schwefel222", "Schwefel221", "Rosenbrock", "Step"
// "Schwefel", "Rastrigin", "Ackley", "Griewank", "Penalized1" 
// "ShiftedSphere", "RotatedElliptic", "Composite1", "Composite2"

// ==================== 參數自動設定 ====================
VisionNetParams getOptimalParams(const string& funcName, int dim) {
    VisionNetParams params;
    
    // 根據維度自動調整網格大小
    if (dim <= 5) params.L = 3;
    else if (dim <= 10) params.L = 5;  
    else if (dim <= 20) params.L = 6;
    else params.L = 7;
    
    params.dimension = dim;
    params.maxEvaluations = dim * 10000;  // 動態調整評估次數
    
    // 根據函數類型調整歷史表大小
    if (funcName == "Sphere" || funcName == "Schwefel222") {
        params.Hsize = 30;  // 單峰函數需要較少歷史
    } else if (funcName == "Ackley" || funcName == "Griewank" || funcName == "Schwefel") {
        params.Hsize = 80;  // 多峰函數需要更多探索
    } else {
        params.Hsize = 50;  // 預設值
    }
    
    return params;
}

// ==================== 中文顯示設定 ====================
void setupChineseDisplay() {
#ifdef _WIN32
    // 設定控制台輸出為UTF-8編碼
    SetConsoleOutputCP(CP_UTF8);
    
    // 設定C++的locale
    setlocale(LC_ALL, ".UTF8");
#endif
}

// ==================== 主程式 ====================
int main() {
    // 設定中文顯示支持
    setupChineseDisplay();
    
    cout << "🚀 Vision Net Algorithm - 簡化測試程式" << endl;
    cout << string(50, '=') << endl;
    
    // 獲取函數資訊
    FunctionInfo funcInfo = TestFunctions::getFunctionInfo(FUNCTION_NAME);
    
    cout << "📊 測試設定：" << endl;
    cout << "  函數名稱：" << funcInfo.name << endl;
    cout << "  函數描述：" << funcInfo.description << endl;
    cout << "  問題維度：" << DIMENSION << "D" << endl;
    cout << "  全域最優值：" << funcInfo.globalOptimum << endl;
    cout << "  搜尋範圍：[" << funcInfo.searchRange[0] << ", " << funcInfo.searchRange[1] << "]^" << DIMENSION << endl;
    
    // 自動設定最佳參數
    VisionNetParams params = getOptimalParams(FUNCTION_NAME, DIMENSION);
    
    // 設定搜尋邊界
    vector<double> lower(DIMENSION, funcInfo.searchRange[0]);
    vector<double> upper(DIMENSION, funcInfo.searchRange[1]);
    params.setBounds(lower, upper);
    
    cout << "\n⚙️ 演算法參數：" << endl;
    cout << "  網格大小：" << params.L << "×" << params.L << " = " << params.L*params.L << " regions" << endl;
    cout << "  最大評估：" << params.maxEvaluations << endl;
    cout << "  歷史表大小：" << params.Hsize << " × " << DIMENSION << " = " << params.Hsize * DIMENSION << endl;
    
    cout << "\n🔄 執行中..." << endl;
    
    // 記錄開始時間
    auto start = chrono::high_resolution_clock::now();
    
    // 建立並執行演算法
    VisionNet vn(params, funcInfo.func);
    vn.RunVN();
    
    // 記錄結束時間
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    // 獲取結果
    GridPoint bestSolution = vn.getBestSolution();
    
    // ==================== 結果顯示 ====================
    cout << "\n" << string(50, '=') << endl;
    cout << "🎯 最佳化結果：" << endl;
    cout << string(50, '=') << endl;
    
    cout << "最佳適應度：" << scientific << setprecision(6) << bestSolution.fitness << endl;
    cout << "誤差（與全域最優）：" << scientific << setprecision(6) << abs(bestSolution.fitness - funcInfo.globalOptimum) << endl;
    cout << "總評估次數：" << vn.getEvaluationCount() << endl;
    cout << "運行時間：" << duration.count() << " ms" << endl;
    cout << "最佳區域：" << bestSolution.gridId << endl;
    
    // 顯示最佳位置（限制顯示長度）
    cout << "最佳位置：[";
    int showDims = min(8, (int)bestSolution.position.size());
    for (int i = 0; i < showDims; i++) {
        cout << fixed << setprecision(4) << bestSolution.position[i];
        if (i < showDims - 1) cout << ", ";
    }
    if (bestSolution.position.size() > showDims) cout << ", ...";
    cout << "]" << endl;
    
    // 收斂品質評估
    double error = abs(bestSolution.fitness - funcInfo.globalOptimum);
    string quality;
    if (error < 1e-10) quality = "🌟 Excellent";
    else if (error < 1e-6) quality = "✅ Very Good"; 
    else if (error < 1e-3) quality = "👍 Good";
    else if (error < 1.0) quality = "👌 Fair";
    else quality = "❌ Poor";
    
    cout << "收斂品質：" << quality << endl;
    
    cout << "\n💡 要測試其他函數，請修改 main.cpp 中的 FUNCTION_NAME 變數" << endl;
    cout << "📋 可用函數：";
    
    vector<string> availableFuncs = TestFunctions::getAvailableFunctions();
    for (int i = 0; i < availableFuncs.size(); i++) {
        if (i % 4 == 0) cout << "\n    ";
        cout << availableFuncs[i];
        if (i < availableFuncs.size() - 1) cout << ", ";
    }
    cout << endl;
    system("pause");
    return 0;
}



