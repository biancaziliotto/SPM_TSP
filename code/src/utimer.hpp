#include <iostream>
#include <iomanip>
#include <chrono>

#define START(timename) auto timename = std::chrono::system_clock::now();
#define STOP(timename, elapsed) auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - timename).count();

class utimer {
private:
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point stop;
    std::string message;
    using usecs = std::chrono::microseconds;
    using msecs = std::chrono::milliseconds;

private:
    long* us_elapsed;

public:
    utimer(const std::string m);
    utimer(const std::string m, long* us);
    ~utimer();
};
