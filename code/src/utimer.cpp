#include "utimer.hpp"

utimer::utimer(const std::string m) : message(m), us_elapsed(nullptr) {
    start = std::chrono::system_clock::now();
}

utimer::utimer(const std::string m, long* us) : message(m), us_elapsed(us) {
    start = std::chrono::system_clock::now();
}

utimer::~utimer() {
    stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    auto musec = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    if (us_elapsed == nullptr)
        std::cout << message << " computed in " << std::setw(15) << musec << " usec "
                  << std::endl;
    if (us_elapsed != nullptr)
        (*us_elapsed) = musec;
}