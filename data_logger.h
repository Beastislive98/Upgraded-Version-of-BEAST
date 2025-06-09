#ifndef DATA_LOGGER_H
#define DATA_LOGGER_H

#include <string>
#include <nlohmann/json.hpp>

// Fix ERROR macro conflict with Windows
#ifdef ERROR
#undef ERROR
#endif

using json = nlohmann::json;

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

// Main logging functions
void log_message(LogLevel level, const std::string& message);
void log_trade(const json& trade_data);
void log_order(const json& order_data);
void log_order(const json& order, const std::string& status);  // Overloaded version
void log_market_data(const json& market_data);

// Specialized logging functions
void log_debug(const std::string& message);
void log_info(const std::string& message);
void log_warning(const std::string& message);
void log_error(const std::string& message);
void log_critical(const std::string& message);

// File management
void rotate_log_files();
void archive_old_logs();
void initialize_logging();
void cleanup_logging();

// Performance logging
void log_execution_time(const std::string& function_name, double execution_time_ms);
void log_api_response_time(const std::string& endpoint, double response_time_ms);

#endif // DATA_LOGGER_H