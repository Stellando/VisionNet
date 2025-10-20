/**
 * functions.cpp - 測試函數集合
 * 包含所有常用的優化測試函數
 * 使用方式：只需在main.cpp中改變函數名稱即可切換測試
 */

#include "functions.h"
#include <cmath>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace TestFunctions {
    
    // ==================== 單峰函數 (Unimodal) ====================
    
    // F1: Sphere 函數
    // f(x) = Σ(xi²)
    // 全域最小值：f(0, 0, ..., 0) = 0
    // 搜尋範圍：[-100, 100]
    double Sphere(const vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += val * val;
        }
        return sum;
    }
    
    // F2: Schwefel's 2.22 函數
    // f(x) = Σ|xi| + Π|xi|
    // 全域最小值：f(0, 0, ..., 0) = 0
    // 搜尋範圍：[-10, 10]
    double Schwefel222(const vector<double>& x) {
        double sum = 0.0;
        double product = 1.0;
        for (double val : x) {
            double absVal = abs(val);
            sum += absVal;
            product *= absVal;
        }
        return sum + product;
    }
    
    // F3: Schwefel's 2.21 函數
    // f(x) = max{|xi|, i=1,2,...,n}
    // 全域最小值：f(0, 0, ..., 0) = 0
    // 搜尋範圍：[-100, 100]
    double Schwefel221(const vector<double>& x) {
        double maxVal = 0.0;
        for (double val : x) {
            maxVal = max(maxVal, abs(val));
        }
        return maxVal;
    }
    
    // F4: Rosenbrock 函數
    // f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²]
    // 全域最小值：f(1, 1, ..., 1) = 0
    // 搜尋範圍：[-30, 30]
    double Rosenbrock(const vector<double>& x) {
        double sum = 0.0;
        for (int i = 0; i < x.size() - 1; i++) {
            double term1 = x[i+1] - x[i] * x[i];
            double term2 = 1.0 - x[i];
            sum += 100.0 * term1 * term1 + term2 * term2;
        }
        return sum;
    }
    
    // F5: Step 函數
    // f(x) = Σ(⌊xi + 0.5⌋)²
    // 全域最小值：f(0, 0, ..., 0) = 0
    // 搜尋範圍：[-100, 100]
    double Step(const vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            double floorVal = floor(val + 0.5);
            sum += floorVal * floorVal;
        }
        return sum;
    }
    
    // ==================== 多峰函數 (Multimodal) ====================
    
    // F6: Schwefel 函數
    // f(x) = 418.9829*D - Σ[xi * sin(√|xi|)]
    // 全域最小值：f(420.9687, ..., 420.9687) = 0
    // 搜尋範圍：[-500, 500]
    double Schwefel(const vector<double>& x) {
        double sum = 0.0;
        for (double xi : x) {
            sum += xi * sin(sqrt(abs(xi)));
        }
        return 418.9829 * x.size() - sum;
    }
    
    // F7: Rastrigin 函數
    // f(x) = 10*n + Σ[xi² - 10*cos(2π*xi)]
    // 全域最小值：f(0, 0, ..., 0) = 0
    // 搜尋範圍：[-5.12, 5.12]
    double Rastrigin(const vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += val * val - 10.0 * cos(2.0 * M_PI * val);
        }
        return 10.0 * x.size() + sum;
    }
    
    // F8: Ackley 函數
    // f(x) = -20*exp(-0.2*√(1/n*Σxi²)) - exp(1/n*Σcos(2π*xi)) + 20 + e
    // 全域最小值：f(0, 0, ..., 0) = 0
    // 搜尋範圍：[-32, 32]
    double Ackley(const vector<double>& x) {
        int n = x.size();
        double sum1 = 0.0, sum2 = 0.0;
        
        for (double val : x) {
            sum1 += val * val;
            sum2 += cos(2.0 * M_PI * val);
        }
        
        double result = -20.0 * exp(-0.2 * sqrt(sum1 / n)) - exp(sum2 / n) + 20.0 + exp(1.0);
        return result;
    }
    
    // F9: Griewank 函數
    // f(x) = 1 + (1/4000)*Σ(xi²) - Π(cos(xi/√i))
    // 全域最小值：f(0, 0, ..., 0) = 0
    // 搜尋範圍：[-600, 600]
    double Griewank(const vector<double>& x) {
        double sum = 0.0;
        double product = 1.0;
        for (int i = 0; i < x.size(); i++) {
            sum += x[i] * x[i];
            product *= cos(x[i] / sqrt(i + 1));
        }
        return 1.0 + sum / 4000.0 - product;
    }
    
    // F10: Penalized 函數 1
    // 複雜的懲罰函數
    // 搜尋範圍：[-50, 50]
    double Penalized1(const vector<double>& x) {
        int n = x.size();
        double sum1 = 0.0;
        
        // 轉換 xi 到 yi = 1 + (xi + 1)/4
        vector<double> y(n);
        for (int i = 0; i < n; i++) {
            y[i] = 1.0 + (x[i] + 1.0) / 4.0;
        }
        
        // 主要部分
        for (int i = 0; i < n - 1; i++) {
            double yi_minus_1 = y[i] - 1.0;
            sum1 += yi_minus_1 * yi_minus_1 * (1.0 + 10.0 * pow(sin(M_PI * y[i+1]), 2));
        }
        
        double yn_minus_1 = y[n-1] - 1.0;
        double result = (M_PI / n) * (10.0 * pow(sin(M_PI * y[0]), 2) + sum1 + yn_minus_1 * yn_minus_1);
        
        // 懲罰項
        for (double xi : x) {
            if (xi > 10.0) result += 100.0 * pow(xi - 10.0, 4);
            else if (xi < -10.0) result += 100.0 * pow(-xi - 10.0, 4);
        }
        
        return result;
    }
    
    // ==================== CEC 風格函數 ====================
    
    // F11: Shifted Sphere 函數
    // f(x) = Σ((xi - oi)²) + fbias
    // 搜尋範圍：[-100, 100]
    double ShiftedSphere(const vector<double>& x) {
        // 預設位移向量（可根據需要調整）
        vector<double> shift = {1.0, 2.0, -1.5, 0.5, -2.0}; // 示例，實際使用時應擴展到x.size()
        
        double sum = 0.0;
        for (int i = 0; i < x.size(); i++) {
            double shifted = x[i] - (i < shift.size() ? shift[i % shift.size()] : 0.0);
            sum += shifted * shifted;
        }
        return sum + 100.0; // fbias = 100
    }
    
    // F12: Rotated High Conditioned Elliptic 函數
    // 高條件數橢圓函數
    // 搜尋範圍：[-100, 100]
    double RotatedElliptic(const vector<double>& x) {
        double sum = 0.0;
        for (int i = 0; i < x.size(); i++) {
            double power = 6.0 * i / (x.size() - 1);
            sum += pow(10.0, power) * x[i] * x[i];
        }
        return sum;
    }
    
    // ==================== 複合函數 ====================
    
    // F13: 複合函數 1 (Sphere + Rastrigin)
    // 搜尋範圍：[-5, 5]
    double Composite1(const vector<double>& x) {
        return 0.5 * Sphere(x) + 0.5 * Rastrigin(x);
    }
    
    // F14: 複合函數 2 (Ackley + Griewank)  
    // 搜尋範圍：[-32, 32]
    double Composite2(const vector<double>& x) {
        return 0.6 * Ackley(x) + 0.4 * Griewank(x);
    }
    
    // ==================== 測試函數資訊 ====================
    
    // 獲取函數資訊
    FunctionInfo getFunctionInfo(const string& funcName) {
        static map<string, FunctionInfo> funcInfoMap = {
            {"Sphere", {"Sphere", Sphere, {-100, 100}, 0.0, "Single-modal, smooth"}},
            {"Schwefel222", {"Schwefel 2.22", Schwefel222, {-10, 10}, 0.0, "Single-modal"}},
            {"Schwefel221", {"Schwefel 2.21", Schwefel221, {-100, 100}, 0.0, "Single-modal, non-smooth"}},
            {"Rosenbrock", {"Rosenbrock", Rosenbrock, {-30, 30}, 0.0, "Narrow curved valley"}},
            {"Step", {"Step", Step, {-100, 100}, 0.0, "Step function, flat regions"}},
            {"Schwefel", {"Schwefel", Schwefel, {-500, 500}, 0.0, "Multi-modal, many local minima"}},
            {"Rastrigin", {"Rastrigin", Rastrigin, {-5.12, 5.12}, 0.0, "Multi-modal, regular structure"}},
            {"Ackley", {"Ackley", Ackley, {-32, 32}, 0.0, "Multi-modal, nearly flat outer region"}},
            {"Griewank", {"Griewank", Griewank, {-600, 600}, 0.0, "Multi-modal, product of cosines"}},
            {"Penalized1", {"Penalized 1", Penalized1, {-50, 50}, 0.0, "Multi-modal with penalty"}},
            {"ShiftedSphere", {"Shifted Sphere", ShiftedSphere, {-100, 100}, 100.0, "Shifted optimum"}},
            {"RotatedElliptic", {"Rotated Elliptic", RotatedElliptic, {-100, 100}, 0.0, "High condition number"}},
            {"Composite1", {"Composite 1", Composite1, {-5, 5}, 0.0, "Sphere + Rastrigin hybrid"}},
            {"Composite2", {"Composite 2", Composite2, {-32, 32}, 0.0, "Ackley + Griewank hybrid"}}
        };
        
        auto it = funcInfoMap.find(funcName);
        if (it != funcInfoMap.end()) {
            return it->second;
        } else {
            return {"Unknown", Sphere, {-100, 100}, 0.0, "Unknown function"};
        }
    }
    
    // 列出所有可用函數
    vector<string> getAvailableFunctions() {
        return {
            "Sphere", "Schwefel222", "Schwefel221", "Rosenbrock", "Step",
            "Schwefel", "Rastrigin", "Ackley", "Griewank", "Penalized1",
            "ShiftedSphere", "RotatedElliptic", "Composite1", "Composite2"
        };
    }
    
} // namespace TestFunctions