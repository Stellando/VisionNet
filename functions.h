/**
 * functions.h - 測試函數標頭檔
 * 包含所有測試函數的聲明和相關資料結構
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <functional>
#include <string>
#include <map>

using namespace std;

// 函數資訊結構
struct FunctionInfo {
    string name;                                    // 函數名稱
    function<double(const vector<double>&)> func;  // 函數指標
    vector<double> searchRange;                     // 搜尋範圍 [lower, upper]
    double globalOptimum;                           // 全域最優值
    string description;                             // 函數描述
};

namespace TestFunctions {
    
    // ==================== 單峰函數 ====================
    double Sphere(const vector<double>& x);        // F1: Sphere 函數
    double Schwefel222(const vector<double>& x);   // F2: Schwefel 2.22
    double Schwefel221(const vector<double>& x);   // F3: Schwefel 2.21
    double Rosenbrock(const vector<double>& x);    // F4: Rosenbrock 函數
    double Step(const vector<double>& x);          // F5: Step 函數
    
    // ==================== 多峰函數 ====================
    double Schwefel(const vector<double>& x);      // F6: Schwefel 函數
    double Rastrigin(const vector<double>& x);     // F7: Rastrigin 函數
    double Ackley(const vector<double>& x);        // F8: Ackley 函數
    double Griewank(const vector<double>& x);      // F9: Griewank 函數
    double Penalized1(const vector<double>& x);    // F10: Penalized 函數
    
    // ==================== CEC 風格函數 ====================
    double ShiftedSphere(const vector<double>& x);    // F11: 位移 Sphere
    double RotatedElliptic(const vector<double>& x);  // F12: 旋轉橢圓
    
    // ==================== 複合函數 ====================
    double Composite1(const vector<double>& x);       // F13: 複合函數 1
    double Composite2(const vector<double>& x);       // F14: 複合函數 2
    
    // ==================== 輔助函數 ====================
    FunctionInfo getFunctionInfo(const string& funcName);  // 獲取函數資訊
    vector<string> getAvailableFunctions();                // 獲取所有可用函數列表
    
} // namespace TestFunctions

#endif // FUNCTIONS_H