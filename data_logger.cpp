// data_logger.cpp - Production Ready Version

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include "data_logger.h"
#include "error_handler.h"

using json = nlohmann::json;

static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

static std::string get_log_filename() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "logs/trading_" << std::put_time(std::localtime(&time_t), "%Y%m%d") << ".log";
    return ss.str();
}

static std::string level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

void initialize_logging() {
    try {
        // Create logs directory if it doesn't exist
        std::filesystem::create_directories("logs");
        std::filesystem::create_directories("logs/archive");
        
        log_info("Logging system initialized");
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize logging: " << e.what() << std::endl;
    }
}

void log_message(LogLevel level, const std::string& message) {
    try {
        std::string timestamp = get_timestamp();
        std::string level_str = level_to_string(level);
        
        std::string log_entry = "[" + timestamp + "] [" + level_str + "] " + message;
        
        // Write to console
        if (level == LogLevel::ERROR || level == LogLevel::CRITICAL) {
            std::cerr << log_entry << std::endl;
        } else {
            std::cout << log_entry << std::endl;
        }
        
        // Write to file
        std::ofstream log_file(get_log_filename(), std::ios::app);
        if (log_file.is_open()) {
            log_file << log_entry << std::endl;
            log_file.close();
        }
        
        // For critical errors, also write to separate critical log
        if (level == LogLevel::CRITICAL) {
            std::ofstream critical_file("logs/critical.log", std::ios::app);
            if (critical_file.is_open()) {
                critical_file << log_entry << std::endl;
                critical_file.close();
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Logging failed: " << e.what() << std::endl;
    }
}

void log_trade(const json& trade_data) {
    try {
        std::string timestamp = get_timestamp();
        
        // Create trade log entry
        json log_entry = {
            {"timestamp", timestamp},
            {"type", "TRADE"},
            {"data", trade_data}
        };
        
        // Write to trades log file
        std::ofstream trade_file("logs/trades.json", std::ios::app);
        if (trade_file.is_open()) {
            trade_file << log_entry.dump() << std::endl;
            trade_file.close();
        }
        
        // Also log to main log
        log_info("TRADE EXECUTED: " + trade_data.dump());
        
    } catch (const std::exception& e) {
        log_error("Failed to log trade: " + std::string(e.what()));
    }
}

void log_order(const json& order_data) {
    try {
        std::string timestamp = get_timestamp();
        
        // Create order log entry
        json log_entry = {
            {"timestamp", timestamp},
            {"type", "ORDER"},
            {"data", order_data}
        };
        
        // Write to orders log file
        std::ofstream order_file("logs/orders.json", std::ios::app);
        if (order_file.is_open()) {
            order_file << log_entry.dump() << std::endl;
            order_file.close();
        }
        
        // Also log to main log
        std::string order_type = order_data.value("side", "UNKNOWN");
        std::string symbol = order_data.value("symbol", "UNKNOWN");
        double quantity = order_data.value("quantity", 0.0);
        
        log_info("ORDER PLACED: " + order_type + " " + std::to_string(quantity) + " " + symbol);
        
    } catch (const std::exception& e) {
        log_error("Failed to log order: " + std::string(e.what()));
    }
}

// Overloaded version with status
void log_order(const json& order, const std::string& status) {
    try {
        std::ofstream file("logs/order_log.txt", std::ios::app);
        if (!file.is_open()) {
            std::cerr << "[ORDER_MANAGER] Failed to open log file." << std::endl;
            return;
        }

        std::time_t now = std::time(nullptr);
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        json entry = {
            {"timestamp", timestamp},
            {"symbol", order.value("symbol", "")},
            {"side", order.value("side", "")},
            {"positionSide", order.value("positionSide", "")},
            {"type", order.value("type", "")},
            {"price", order.value("price", 0.0)},
            {"quantity", order.value("quantity", 0.0)},
            {"stopPrice", order.value("stopPrice", 0.0)},
            {"status", status},
            {"reduceOnly", order.value("reduceOnly", false)}
        };
        
        // Add orderId if present
        if (order.contains("orderId")) {
            entry["orderId"] = order["orderId"];
        }
        
        // Add clientOrderId if present
        if (order.contains("clientOrderId")) {
            entry["clientOrderId"] = order["clientOrderId"];
        }

        file << entry.dump(2) << "\n\n";
        file.close();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to log order: " << e.what() << std::endl;
    }
}

void log_market_data(const json& market_data) {
    try {
        std::string timestamp = get_timestamp();
        
        // Create market data log entry
        json log_entry = {
            {"timestamp", timestamp},
            {"type", "MARKET_DATA"},
            {"data", market_data}
        };
        
        // Write to market data log file
        std::ofstream market_file("logs/market_data.json", std::ios::app);
        if (market_file.is_open()) {
            market_file << log_entry.dump() << std::endl;
            market_file.close();
        }
        
    } catch (const std::exception& e) {
        log_error("Failed to log market data: " + std::string(e.what()));
    }
}

void log_debug(const std::string& message) {
    log_message(LogLevel::DEBUG, message);
}

void log_info(const std::string& message) {
    log_message(LogLevel::INFO, message);
}

void log_warning(const std::string& message) {
    log_message(LogLevel::WARNING, message);
}

void log_error(const std::string& message) {
    log_message(LogLevel::ERROR, message);
}

void log_critical(const std::string& message) {
    log_message(LogLevel::CRITICAL, message);
}

void log_execution_time(const std::string& function_name, double execution_time_ms) {
    log_info("PERFORMANCE: " + function_name + " executed in " + 
             std::to_string(execution_time_ms) + "ms");
}

void log_api_response_time(const std::string& endpoint, double response_time_ms) {
    log_info("API_RESPONSE: " + endpoint + " responded in " + 
             std::to_string(response_time_ms) + "ms");
}

void rotate_log_files() {
    try {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        // Archive files older than 30 days
        auto cutoff_time = now - std::chrono::hours(24 * 30);
        
        for (const auto& entry : std::filesystem::directory_iterator("logs")) {
            if (entry.is_regular_file()) {
                auto file_time = std::filesystem::last_write_time(entry);
                auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    file_time - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                
                if (sctp < cutoff_time) {
                    // Move to archive folder
                    std::string archive_dir = "logs/archive";
                    if (!std::filesystem::exists(archive_dir)) {
                        std::filesystem::create_directory(archive_dir);
                    }
                    
                    std::filesystem::path old_path = entry.path();
                    std::filesystem::path new_path = archive_dir + "/" + old_path.filename().string();
                    std::filesystem::rename(old_path, new_path);
                    
                    log_info("Archived old log file: " + old_path.filename().string());
                }
            }
        }
        
    } catch (const std::exception& e) {
        log_error("Failed to rotate log files: " + std::string(e.what()));
    }
}

void archive_old_logs() {
    try {
        std::string archive_dir = "logs/archive";
        if (!std::filesystem::exists(archive_dir)) {
            std::filesystem::create_directory(archive_dir);
        }
        
        auto now = std::chrono::system_clock::now();
        auto yesterday = now - std::chrono::hours(24);
        
        for (const auto& entry : std::filesystem::directory_iterator("logs")) {
            if (entry.is_regular_file() && entry.path().extension() == ".log") {
                auto file_time = std::filesystem::last_write_time(entry);
                auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    file_time - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                
                if (sctp < yesterday) {
                    std::filesystem::path old_path = entry.path();
                    std::filesystem::path new_path = archive_dir + "/" + old_path.filename().string();
                    std::filesystem::copy_file(old_path, new_path);
                    std::filesystem::remove(old_path);
                    
                    log_info("Archived log file: " + old_path.filename().string());
                }
            }
        }
        
    } catch (const std::exception& e) {
        log_error("Failed to archive logs: " + std::string(e.what()));
    }
}

void cleanup_logging() {
    log_info("Logging system shutting down");
    
    // Perform final log rotation
    rotate_log_files();
}

bool archive_trade_data(const json& trade, const std::string& filename_prefix) {
    try {
        // Ensure logs directory exists
        std::filesystem::create_directories("logs");
        
        // Generate timestamp
        std::time_t now = std::time(nullptr);
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));

        // Create file path
        std::string filename = "logs/" + filename_prefix + "_" + std::string(timestamp) + ".json";

        // Open file
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[DATA_LOGGER] Failed to write trade archive to " << filename << std::endl;
            log_error("Failed to open file for archive: " + filename);
            return false;
        }

        // Write JSON with pretty printing
        file << trade.dump(2);
        file.close();
        
        std::cout << "[DATA_LOGGER] Archived trade data to " << filename << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[DATA_LOGGER] Error archiving trade data: " << e.what() << std::endl;
        log_error("Failed to archive trade data: " + std::string(e.what()));
        return false;
    }
}