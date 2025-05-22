#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <related.hpp>



int main (void) {
    // 示例数据
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> y = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}; // 完全正相关

    try {
        double pearson = related::pearson(x, y);
        std::cout << "Pearson correlation coefficient: " << pearson << std::endl;
        
        // 测试负相关
        std::vector<double> y_neg = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        pearson = related::pearson(x, y_neg);
        std::cout << "Negative correlation: " << pearson << std::endl;
        
        // 测试无相关
        std::vector<double> y_no = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
        pearson = related::pearson(x, y_no);
        std::cout << "No correlation: " << pearson << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}