#include "functions.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

// ==================== CEC2022 全域變數定義 ====================
// 這些變數是 cec22_test_func.cpp 所需要的
double *OShift = nullptr;
double *M = nullptr;
double *y = nullptr;
double *z = nullptr;
double *x_bound = nullptr;
int ini_flag = 0;
int n_flag = 0;
int func_flag = 0;
int *SS = nullptr;

// 宣告外部 CEC2022 函數
void cec22_test_func(double *x, double *f, int nx, int mx, int func_num);

namespace CEC22Functions {

    // 執行 CEC2022 評估的包裝函數
    double evaluate(int funcId, const vector<double>& x) {
        double f = 0.0;
        // 複製輸入向量，因為 cec22_test_func 接受 double* 且可能在內部進行操作
        // 雖然通常輸入不會被修改，但為了安全起見使用副本
        vector<double> x_copy = x; 
        
        // 呼叫 CEC2022 測試函數
        // nx = x.size(), mx = 1 (一次評估一個解)
        cec22_test_func(x_copy.data(), &f, (int)x.size(), 1, funcId);
        
        return f;
    }

    // 獲取函數資訊
    FunctionInfo getFunctionInfo(int funcId, int dimension) {
        FunctionInfo info;
        info.name = "CEC2022 F" + to_string(funcId);
        info.searchRange = {-100.0, 100.0}; // CEC2022 預設搜尋範圍 [-100, 100]
        
        // 設定全域最優值 (Bias)
        // 根據 CEC2022 定義，每個函數都有一個偏移值 (Bias)
        // 演算法的目標是讓 f(x) - Bias = 0
        double bias = 0.0;
        switch(funcId) {
            case 1: bias = 300.0; break;
            case 2: bias = 400.0; break;
            case 3: bias = 600.0; break;
            case 4: bias = 800.0; break;
            case 5: bias = 900.0; break;
            case 6: bias = 1800.0; break;
            case 7: bias = 2000.0; break;
            case 8: bias = 2200.0; break;
            case 9: bias = 2300.0; break;
            case 10: bias = 2400.0; break;
            case 11: bias = 2600.0; break;
            case 12: bias = 2700.0; break;
            default: bias = 0.0; break;
        }
        info.globalOptimum = bias;
        
        // 設定描述
        switch(funcId) {
            case 1: info.description = "Shifted and full Rotated Zakharov Function"; break;
            case 2: info.description = "Shifted and full Rotated Rosenbrock's Function"; break;
            case 3: info.description = "Shifted and full Rotated Expanded Schaffer's f6 Function"; break;
            case 4: info.description = "Shifted and full Rotated Non-Continuous Rastrigin's Function"; break;
            case 5: info.description = "Shifted and full Rotated Levy Function"; break;
            case 6: info.description = "Hybrid Function 1 (N=3)"; break;
            case 7: info.description = "Hybrid Function 2 (N=6)"; break;
            case 8: info.description = "Hybrid Function 3 (N=5)"; break;
            case 9: info.description = "Composition Function 1 (N=5)"; break;
            case 10: info.description = "Composition Function 2 (N=4)"; break;
            case 11: info.description = "Composition Function 3 (N=5)"; break;
            case 12: info.description = "Composition Function 4 (N=6)"; break;
            default: info.description = "Unknown Function"; break;
        }
        
        // 綁定函數 ID 到 Lambda
        info.func = [funcId](const vector<double>& x) {
            return evaluate(funcId, x);
        };
        
        return info;
    }
}