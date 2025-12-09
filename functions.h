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
#include "structure.h"

using namespace std;

// 函數資訊結構
struct FunctionInfo {
    string name;                                    // 函數名稱
    function<double(const vector<double>&)> func;  // 函數指標
    vector<double> searchRange;                     // 搜尋範圍 [lower, upper]
    double globalOptimum;                           // 全域最優值
    string description;                             // 函數描述
};

namespace CEC22Functions {
    // 獲取 CEC2022 函數資訊
    FunctionInfo getFunctionInfo(int funcId, int dimension);
    
    // 執行 CEC2022 評估
    double evaluate(int funcId, const vector<double>& x);
}

#endif // FUNCTIONS_H