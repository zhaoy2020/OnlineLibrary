#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <related.hpp>



// Pearson correlation.
double related::pearson(const std::vector<double>& x, const std::vector<double>& y) {
    // 检查输入数据长度是否相同
    if (x.size() != y.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }
    
    size_t n = x.size();
    if (n == 0) {
        throw std::invalid_argument("Input vectors cannot be empty");
    }

    // 计算各项和
    double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    double sum_x_sq = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    double sum_y_sq = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
    double sum_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);

    // 计算分子和分母
    double numerator = sum_xy - (sum_x * sum_y / n);
    double denominator_x = sum_x_sq - (sum_x * sum_x / n);
    double denominator_y = sum_y_sq - (sum_y * sum_y / n);
    // 防止除以零
    if (denominator_x <= 0 || denominator_y <= 0) {
        return 0.0; // 无相关性
    }

    // 计算Pearson相关系数
    double r = numerator / std::sqrt(denominator_x * denominator_y);
    
    // 确保结果在[-1, 1]范围内
    return std::max(-1.0, std::min(1.0, r));
}


// Spearman correlation.
double related::spearman(double a, double b) {
    return 0.0;
}