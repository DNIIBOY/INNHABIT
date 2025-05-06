#ifndef MACROS_H
#define MACROS_H

#include <iostream>
// Logging helpers
#define LOG(msg) std::cout << "[INFO] " << msg << endl
#define ERROR(msg) std::cerr << "[ERROR] " << msg << endl
//                 return nullptr;

#endif // MACROS_H