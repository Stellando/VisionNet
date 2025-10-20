#include "structure.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==================== HistoryTable 實作 ====================

void HistoryTable::addParameter(const HistoryParameter& param) {
    parameters.push_back(param);
    
    // 如果超過最大大小，移除最舊的項目
    if (parameters.size() > maxSize) {
        parameters.erase(parameters.begin());
    }
}

HistoryParameter HistoryTable::getBestParameter(int dim) const {
    HistoryParameter bestParam(0.3, 0.3, dim, 0);  // 預設值
    
    if (parameters.empty()) return bestParam;
    
    double bestPerformance = -numeric_limits<double>::infinity();
    for (const auto& param : parameters) {
        if (param.dimension == dim && param.performance > bestPerformance) {
            bestPerformance = param.performance;
            bestParam = param;
        }
    }
    
    return bestParam;
}

vector<HistoryParameter> HistoryTable::getRecentParameters(int count) const {
    vector<HistoryParameter> recentParams;
    
    if (parameters.empty()) return recentParams;
    
    int actualCount = min(count, (int)parameters.size());
    int startIndex = max(0, (int)parameters.size() - actualCount);
    
    for (int i = startIndex; i < parameters.size(); i++) {
        recentParams.push_back(parameters[i]);
    }
    
    return recentParams;
}

void HistoryTable::updatePerformance(int index, double performance) {
    if (index >= 0 && index < parameters.size()) {
        parameters[index].performance = performance;
    }
}

void HistoryTable::clear() {
    parameters.clear();
}

void HistoryTable::updateMCr(int index, double newMCr) {
    if (index >= 0 && index < parameters.size()) {
        parameters[index].MCr = newMCr;
    }
}

void HistoryTable::updateMF(int index, double newMF) {
    if (index >= 0 && index < parameters.size()) {
        parameters[index].MF = newMF;
    }
}

// ==================== 適應度函數實作 ====================

namespace FitnessFunctions {
    
    // Sphere 函數（最小化問題）
    double Sphere(const vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += val * val;
        }
        return sum;  // 最小值為 0（在原點）
    }
    
    // Ackley 函數（最小化問題）
    double Ackley(const vector<double>& x) {
        int n = x.size();
        double sum1 = 0.0, sum2 = 0.0;
        
        for (double val : x) {
            sum1 += val * val;
            sum2 += cos(2.0 * M_PI * val);
        }
        
        double result = -20.0 * exp(-0.2 * sqrt(sum1 / n)) - exp(sum2 / n) + 20.0 + exp(1.0);
        return result;  // 最小值為 0（在原點）
    }
    
    // Rastrigin 函數（最小化問題）
    double Rastrigin(const vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += val * val - 10.0 * cos(2.0 * M_PI * val);
        }
        return 10.0 * x.size() + sum;  // 最小值為 0（在原點）
    }
    
    // OneMax 問題：計算二進制字串中 1 的個數（最大化問題，用負值轉為最小化）
    double OneMax(const vector<double>& x) {
        double count = 0.0;
        for (double val : x) {
            if (val >= 0.5) {  // 將實數值轉換為二進制
                count += 1.0;
            }
        }
        return -count;  // 轉為最小化問題（最小值為 -n）
    }
}