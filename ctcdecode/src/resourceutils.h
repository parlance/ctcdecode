// #pragma once

// #include <unistd.h>
// #include <unordered_map>
// #include <vector>

// #include "logger.h"

// class ProcessInfo {
// public:
//     /** @brief Find the Nth percentile value from the provided vector
//      * @param values A sorted vector of any datatype
//      * @param percentile a float value between 0 and 1.
//      */
//     template <typename T>
//     static double Percentile(std::vector<T> const& values, float percentile)
//     {
//         double position = ((double)values.size() - 1.0) * percentile;
//         int positionFloor = std::floor(position);
//         if (position == positionFloor)
//             return (double)values[positionFloor];
//         double fractionValue = std::fmod(position, (double)positionFloor);
//         double value
//             = (double)values[positionFloor]
//               + (fractionValue * (double)(values[positionFloor + 1] - values[positionFloor]));
//         return value;
//     }
//     static std::string GetInfo();
//     static std::string FormatProcessInfoVector(
//         const std::vector<std::unordered_map<std::string, double>>& processInfo);

//     int fdCount, pid, tid;
//     double cpuPercent, memoryPercent, vsz, rss, framesProcessed, avgLatency;
//     double rtf, rts, latency90P, latency97P, latency99P, latency, batchSize;
//     double sampleRate;

//     ProcessInfo();

//     void CalculateAvgStats();
//     inline void SortLatencyValues();
//     inline void PushToLatencyValues(double latency);

// private:
//     static std::string Prd(const double x, int decDigits, const int width);
//     static std::string Center(const std::string s, const int w);

//     std::vector<double> latencyValues;

//     void CalculateRealTimeValues();
//     void CalculateAvgResources();
//     void CalculatePercentile();
// };

// class ResourceUtil {
// public:
//     ResourceUtil();

//     std::unordered_map<std::string, double> GetAverageStats();
//     ProcessInfo GetResourceValues(pid_t pid);
//     void Monitor(double latency = 0);
//     void SetBatchSize(int batchSize);
//     void SetSampleRate(std::string sampRate);
//     void SetLogger(Logger* logger);
//     void SetPid(pid_t pid);

// private:
//     int timePosition, vszPosition;
//     double clockTicks, pageSizeKB;
//     std::string directoryStr, fdCommand;
//     ProcessInfo processInfo;
//     timespec tSleep;
//     Logger* logger;

//     int FromShell();
//     double GetMemoryPercent(double rss);
//     std::string GetDateAndTime();
//     void Init();
//     void ReadStatFile(double* vsz, double* rss, double* utime, double* stime);
//     void ResourceCalculator(double* vsz, double* rss, double* cpuPercent);
//     void SetDirectoryStrings();
//     void
//     WriteToFile(double vsz, double memoryPercent, double cpuPercent, int fdCount, double
//     latency); void WriteToLogger(double vsz, double memoryPercent, double cpuPercent, int
//     fdCount, double latency);
// };

// /*
// Usage:
//     ResourceUtil resourceUtil;

//     // this method will calculate CPU%, RAM%, VM(kb), filecount

//     // your function
//     resourceUtil.CalculateResource();

//     // if provided with latency per frame, it will also calculate
//     // avg time per latency and total latency time taken
//     resourceUtil.CalculateResource(latency_value);

//     // additionally you can set samplerate and input batchsize of your model.
//     resourceUtil.SetSampleRate(16000);
//     resourceUtil.SetBatchSize(1024);
//     // the above values will be used to calculate the RTS and percentile.

//     // Finally after processing call
//     resourceUtil.GetAverageStats();
//     // this will return an unordered_map of string and double, regarding the
//     // stats monitored.

// */
