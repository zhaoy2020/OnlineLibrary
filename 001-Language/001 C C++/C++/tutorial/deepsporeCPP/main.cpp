#include <iostream>
#include <cal.hpp>


int main() {
    double x = 2.5; 
    double y = 3.5;
    std::cout << x << "+" << y << "=" << cal::add(x, y) << std::endl;
    return 0;
}

// Method one:
// g++ -c cal.cpp  -I ../include -o cal.o
// g++ test.cpp cal.o -I ../include -o test
// ./test


// Method two:
// math_utils.cpp  --(g++ -c)--> math_utils.o --+
//                                              |--(g++ linker)--> myapp
// main.cpp       --(g++ -c)--> main.o      ----+

// 1. 分离:预处理 --> 编译 (.i) --> 汇编 为机器码 (.o)
// g++ -c cal.cpp -I ../include -o cal.o
// g++ -c test.cpp -I ../include -o test.o

// 2. 链接目标文件生成可执行文件
// g++ cal.o  test.o -o test
// ./test