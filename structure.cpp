#include "structure.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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