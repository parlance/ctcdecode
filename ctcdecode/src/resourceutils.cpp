// #include <algorithm>
// #include <cmath>
// #include <dirent.h>
// #include <iomanip>
// #include <iostream>
// #include <sys/syscall.h>

// #include "resourceutils.h"

// #define FORMATINFO(stream, key, value)                                                             \
//     (stream << std::left << std::setw(17) << std::setfill(' ') << key << value)

// ProcessInfo::ProcessInfo()
// {
//     pid = getpid();
//     tid = syscall(SYS_gettid);
//     fdCount = 0;
//     cpuPercent = 0.0;
//     memoryPercent = 0.0;
//     vsz = 0.0;
//     rss = 0.0;
//     framesProcessed = 0;
//     latency = 0.0;
// }

// /** @brief calculates rtf, rts */
// void ProcessInfo::CalculateRealTimeValues()
// {
//     rtf = latency / ((framesProcessed * batchSize) / sampleRate);
//     rtf /= 1000.0; // rtf will be in ms, converting it to seconds
//     rts = 1.0 / rtf;
// }

// /** @brief calculates the avg vsz, rss, ram%, cpu%, latency */
// void ProcessInfo::CalculateAvgResources()
// {
//     vsz /= framesProcessed;
//     rss /= framesProcessed;
//     memoryPercent /= framesProcessed;
//     cpuPercent /= framesProcessed;
//     avgLatency = latency / framesProcessed;
//     latency /= 1000.0;
// }

// /** @brief Calculates the 90th, 97th, 99th Percentile values from the latencyValues vector */
// void ProcessInfo::CalculatePercentile()
// {
//     latency90P = Percentile(latencyValues, 0.9);
//     latency97P = Percentile(latencyValues, 0.97);
//     latency99P = Percentile(latencyValues, 0.99);
// }

// /** @brief calculates the consolidates average stats of the process information */
// void ProcessInfo::CalculateAvgStats()
// {
//     CalculateRealTimeValues();
//     CalculateAvgResources();
//     CalculatePercentile();
//     latencyValues.clear();
// }

// /** @brief sorts the latencyValues vector */
// void ProcessInfo::SortLatencyValues() { std::sort(latencyValues.begin(), latencyValues.end()); }

// /** @brief pushes the latency value into the latencyValues vector */
// void ProcessInfo::PushToLatencyValues(double latency) { latencyValues.push_back(latency); }

// ResourceUtil::ResourceUtil()
//     : logger(nullptr)
// {
//     Init();
// }

// /** @brief   get the logger object to log resources
//  *  @param   logger logger object pointer
//  *  @returns void
//  */
// void ResourceUtil::SetLogger(Logger* logger) { this->logger = logger; }

// /** @brief set the directory string to parse the stat file and filecount value */
// void ResourceUtil::SetDirectoryStrings()
// {
//     std::string pidStr = std::to_string(processInfo.pid);
//     std::string tidStr = std::to_string(processInfo.tid);
//     DIR* dir = opendir(("/proc/" + tidStr).c_str());
//     if (dir == NULL) {
//         directoryStr = "/proc/" + pidStr + "/task/" + tidStr + "/stat";
//         fdCommand = "find /proc/" + pidStr + "/task/" + tidStr + "/fd | wc -l";
//     } else {
//         directoryStr = "/proc/" + tidStr + "/stat";
//         fdCommand = "find /proc/" + tidStr + "/fd | wc -l";
//         closedir(dir);
//     }
// }

// /** @brief set the batchsize of the model, this value is later used
//  * to compute the RTF and RTS value
//  */
// void ResourceUtil::SetBatchSize(int batchSize) { processInfo.batchSize = (double)batchSize; }

// void ResourceUtil::SetSampleRate(std::string sampleRate)
// {
//     processInfo.sampleRate = (double)std::stoi(sampleRate);
// }

// /** @brief monitors the provided pid process */
// void ResourceUtil::SetPid(pid_t pid)
// {
//     processInfo.pid = pid;
//     processInfo.tid = pid;
//     SetDirectoryStrings();
// }

// /** @brief   initialises constant values
//  *  @returns void
//  */
// void ResourceUtil::Init()
// {
//     tSleep.tv_sec = 0;
//     tSleep.tv_nsec = 10000000L;                   // 10ms
//     clockTicks = sysconf(_SC_CLK_TCK);            // Eg: 100
//     pageSizeKB = sysconf(_SC_PAGE_SIZE) / 1024.0; // Eg: 4
//     timePosition = 14;
//     vszPosition = 23;
//     SetDirectoryStrings();
// }

// /** @brief   get the current date and time string
//  *  @returns std::string
//  */
// std::string ResourceUtil::GetDateAndTime()
// {
//     time_t now;
//     time(&now);
//     struct tm tstruct = *localtime(&now);
//     std::ostringstream currTime;
//     currTime << tstruct.tm_year + 1900 << "-" << tstruct.tm_mon << "-" << tstruct.tm_mday << " ";
//     currTime << tstruct.tm_hour << ":" << tstruct.tm_min << ":" << tstruct.tm_sec;
//     return currTime.str();
// }

// /** @brief   reads the vsz, rss, stime and utime of the process from the stat file
//  *  @param   vsz         virtual memory
//  *  @param   rss         RAM memory percentage
//  *  @param   utime       user time value
//  *  @param   stime       system time[kernel time]
//  *  @returns void
//  *
//  *  From the stat file, the value
//  *      vsz     - will be in bytes - Eg: 175415296
//  *      rss     - will be number of pages in memory - Eg: 24311
//  *      utime   - will be in clock ticks - Eg: 244
//  *      stime   - will be in clock ticks - Eg: 13
//  */
// void ResourceUtil::ReadStatFile(double* vsz, double* rss, double* utime, double* stime)
// {
//     int count = 1;
//     std::string resourceValue;
//     std::ifstream statStream(directoryStr, std::ios_base::in);
//     while (statStream.good()) {
//         if (count == timePosition) {
//             statStream >> (*utime) >> (*stime);
//             if (!vsz) {
//                 break;
//             }
//             count += 2;
//         } else if (count == vszPosition) {
//             statStream >> (*vsz) >> (*rss);
//             break;
//         } else {
//             statStream >> resourceValue;
//             count++;
//         }
//     }
//     statStream.close();
// }

// /** @brief   returns the filecount of the process
//  *  @returns int
//  */
// int ResourceUtil::FromShell()
// {
//     FILE* pipe = popen(fdCommand.c_str(), "r");
//     if (pipe == nullptr) {
//         return -1;
//     }
//     char buf[128];
//     while (fgets(buf, 128, pipe) != nullptr) {
//         int count = std::stoi(buf);
//         pclose(pipe);
//         return count - 1;
//     }
//     if (ferror(pipe)) {
//         return -1;
//     }
//     if (pclose(pipe) == -1) {
//         return -1;
//     }
//     return -2;
// }

// /** @brief   returns the memory usage percentage of the process
//  *  @param   rss     Resident set size value
//  *  @returns double
//  */
// double ResourceUtil::GetMemoryPercent(double rss)
// {
//     double total, mem;
//     std::string resourceValue;
//     std::ifstream stream("/proc/meminfo", std::ios_base::in);
//     while (stream.good()) {
//         stream >> resourceValue;
//         if (resourceValue.find("MemTotal") != std::string::npos) {
//             stream >> total;
//             break;
//         }
//     }
//     stream.close();
//     mem = (rss / total) * 100;
//     return rss;
// }

// /** @brief   calculates the vsz, rss, cpupercent and of the process
//  *  @param   vsz             virtual memory
//  *  @param   memoryPercent      RAM memory percentage
//  *  @param   cpuPercent      CPU percentage
//  *  @returns void
//  */
// void ResourceUtil::ResourceCalculator(double* vsz, double* rss, double* cpuPercent)
// {
//     double proc, time;
//     double utime1, utime2, stime1, stime2;
//     timespec tStart, tEnd, tRemaining;
//     clock_gettime(CLOCK_MONOTONIC, &tStart);
//     tStart.tv_nsec = tStart.tv_nsec;
//     ReadStatFile(nullptr, nullptr, &utime1, &stime1);
//     int val = nanosleep(&tSleep, &tRemaining);
//     if (val == -1) {
//         nanosleep(&tRemaining, nullptr);
//     }
//     clock_gettime(CLOCK_MONOTONIC, &tEnd);
//     tEnd.tv_nsec = tEnd.tv_nsec;
//     ReadStatFile(vsz, rss, &utime2, &stime2);
//     proc = ((utime2 - utime1) + (stime2 - stime1)) / clockTicks;
//     time = ((double)(tEnd.tv_nsec - tStart.tv_nsec)) * 1e-9; // converting nano_second to second
//     *cpuPercent = ((proc / time) * 100);
// }

// /** @brief   log the resource values calculated, to the logger object provided
//  *  @param   vsz         virtual memory
//  *  @param   memoryPercent  RAM memory percentage
//  *  @param   cpuPercent  CPU percentage
//  *  @param   fdCount     Files count
//  *  @param   latency     Latency time for processing this frame
//  *  @returns void
//  */
// void ResourceUtil::WriteToLogger(double vsz,
//                                  double memoryPercent,
//                                  double cpuPercent,
//                                  int fdCount,
//                                  double latency)
// {
//     if (latency != 0) {
//         logger->Log(LogLevel::DEBUG,
//                     "PID:",
//                     processInfo.pid,
//                     ", TID: ",
//                     processInfo.tid,
//                     ", VSZ_VALUE:",
//                     vsz,
//                     "kb, MEMORY_PERCENT:",
//                     memoryPercent,
//                     "%, CPU_PERCENT:",
//                     cpuPercent,
//                     "%, FILES_COUNT:",
//                     fdCount,
//                     "FRAME_LATENCY:",
//                     latency,
//                     "ms");
//     } else {
//         logger->Log(LogLevel::DEBUG,
//                     "PID:",
//                     processInfo.pid,
//                     ", TID: ",
//                     processInfo.tid,
//                     ", VSZ_VALUE:",
//                     vsz,
//                     "kb, MEMORY_PERCENT:",
//                     memoryPercent,
//                     "%, CPU_PERCENT:",
//                     cpuPercent,
//                     "%, FILES_COUNT:",
//                     fdCount);
//     }
// }

// /** @brief   writes the resource values calculated to the resource.txt file
//  *  @param   vsz         virtual memory
//  *  @param   memoryPercent  RAM memory percentage
//  *  @param   cpuPercent  CPU percentage
//  *  @param   fdCount     Files count
//  *  @param   latency     Latency time for processing this frame
//  *  @returns void
//  */
// void ResourceUtil::WriteToFile(double vsz,
//                                double memoryPercent,
//                                double cpuPercent,
//                                int fdCount,
//                                double latency)
// {
//     std::ofstream file("resource.txt", std::ios_base::app);
//     std::string date_time = GetDateAndTime();
//     file << date_time << ' ' << "PID: " << processInfo.pid << ", ";
//     file << "TID: " << processInfo.tid << ", ";
//     file << "VSZ_VALUE: " << vsz << " kb, ";
//     file << "MEMORY_PERCENT: " << memoryPercent << " kb, ";
//     file << "CPU_PERCENT: " << cpuPercent << " %, ";
//     file << "FILES_COUNT: " << fdCount;
//     if (latency != 0) {
//         file << " , FRAME_LATENCY: " << latency << "ms";
//     }
//     file << std::endl;
//     file.close();
// }

// /** @brief   returns the average resource values calculated so far, as string
//  *  @returns void
//  */
// std::unordered_map<std::string, double> ResourceUtil::GetAverageStats()
// {
//     std::unordered_map<std::string, double> processDetails;
//     processInfo.SortLatencyValues();
//     processInfo.CalculateAvgStats();
//     processDetails["FRAMES"] = processInfo.framesProcessed;
//     processDetails["TOT_LAT"] = processInfo.latency;
//     processDetails["AVG_LAT"] = processInfo.avgLatency;
//     // processDetails["99P"] = processInfo.latency99P;
//     // processDetails["97P"] = processInfo.latency97P;
//     // processDetails["90P"] = processInfo.latency90P;
//     processDetails["AVG_CPU"] = processInfo.cpuPercent;
//     processDetails["AVG_RAM"] = processInfo.memoryPercent;
//     processDetails["AVG_VSZ"] = processInfo.vsz;
//     processDetails["RTS"] = processInfo.rts;
//     processDetails["TID"] = processInfo.tid;
//     processDetails["PID"] = processInfo.pid;
//     return processDetails;
// }

// /** @brief   calculates the resource details of pid and write it to file
//  *  @param   latency     Latency time taken to process a frame
//  *  @return  void
//  */
// void ResourceUtil::Monitor(double latency)
// {
//     double vsz, rss, cpuPercent, memoryPercent;
//     ResourceCalculator(&vsz, &rss, &cpuPercent);
//     vsz = vsz / 1024.0;
//     rss = rss * pageSizeKB;
//     memoryPercent = GetMemoryPercent(rss);
//     processInfo.fdCount = FromShell();
//     latency *= 1000;
//     if (logger) {
//         WriteToLogger(vsz, memoryPercent, cpuPercent, processInfo.fdCount, latency);
//     } else {
//         WriteToFile(vsz, memoryPercent, cpuPercent, processInfo.fdCount, latency);
//     }
//     processInfo.cpuPercent += cpuPercent;
//     processInfo.memoryPercent += memoryPercent;
//     processInfo.vsz += vsz;
//     processInfo.rss += rss;
//     processInfo.framesProcessed++;
//     processInfo.latency += latency;
//     processInfo.PushToLatencyValues(latency);
// }

// /** @brief gets the current resource consumption values of the pid */
// ProcessInfo ResourceUtil::GetResourceValues(pid_t pid)
// {
//     double vsz, rss, cpuPercent;
//     processInfo.pid = pid;
//     std::string pidStr = std::to_string(pid);
//     directoryStr = "/proc/" + pidStr + "/stat";
//     fdCommand = "find /proc/" + pidStr + "/fd | wc -l";
//     ResourceCalculator(&vsz, &rss, &cpuPercent);
//     processInfo.vsz = vsz / 1024.0;
//     processInfo.rss = rss * pageSizeKB;
//     processInfo.memoryPercent = GetMemoryPercent(processInfo.rss);
//     processInfo.fdCount = FromShell();
//     processInfo.cpuPercent = cpuPercent;
//     return processInfo;
// }

// /** @brief aligning function used by GetHeaders */
// std::string ProcessInfo::Center(const std::string s, const int w)
// {
//     std::ostringstream ss, spaces;
//     int padding = w - s.size();
//     for (int i = 0; i < padding / 2; ++i)
//         spaces << " ";
//     ss << spaces.str() << s << spaces.str();
//     if (padding > 0 && padding % 2 != 0)
//         ss << " ";
//     return ss.str();
// }

// /** @brief aligning function used by FormatToString */
// std::string ProcessInfo::Prd(const double x, int decDigits, const int width)
// {
//     if (x == std::floor(x))
//         decDigits = 0;
//     std::stringstream ss;
//     ss << std::fixed << std::right;
//     ss.fill(' ');
//     ss.width(width);
//     ss.precision(decDigits);
//     ss << x;
//     return ss.str();
// }

// /** @brief Explain each stat keys */
// std::string ProcessInfo::GetInfo()
// {
//     std::ostringstream info;
//     info << "\n";
//     FORMATINFO(info, "PID", ": Process ID\n");
//     FORMATINFO(info, "TID", ": Thread ID\n");
//     FORMATINFO(info, "AVG_VSZ", ": Average Virtual Memory size of the  (in kb)\n");
//     FORMATINFO(info, "AVG_RAM", ": Average RAM usage of the process (in %)\n");
//     FORMATINFO(info, "AVG_CPU", ": Average CPU usage of the thread (in %)\n");
//     FORMATINFO(info,
//                "RTS",
//                ": 1 / RTF (Real Time Speed) - how much the synthesis is faster than realtime. "
//                "If the value you get is 5 then it means, the library takes 1second to denoise a "
//                "5second input.\n");
//     FORMATINFO(info, "AVG_LAT", ": Average processing time for a frame (in milli seconds)\n");
//     FORMATINFO(info, "TOT_LAT", ": Sum of processing time of frames (in seconds)\n");
//     FORMATINFO(info, "FRAMES", ": Total number of frames processed\n");
//     return info.str();
// }

// /** @brief Formats the provided vector of process infos into a string */
// std::string ProcessInfo::FormatProcessInfoVector(
//     const std::vector<std::unordered_map<std::string, double>>& processInfos)
// {
//     std::ostringstream finalStats;
//     std::for_each(processInfos[0].cbegin(),
//                   processInfos[0].cend(),
//                   [&finalStats](const std::pair<std::string, double>& pair) {
//                       finalStats << Center(pair.first, 12) << " | ";
//                   });
//     finalStats << "\n";
//     for (const std::unordered_map<std::string, double>& processInfo : processInfos) {
//         std::for_each(processInfo.cbegin(),
//                       processInfo.cend(),
//                       [&finalStats](const std::pair<std::string, double>& pair) {
//                           finalStats << Prd(pair.second, 4, 12) << " | ";
//                       });
//         finalStats << "\n";
//     }
//     finalStats << ProcessInfo::GetInfo();
//     return finalStats.str();
// }

// #undef FORMATINFO
