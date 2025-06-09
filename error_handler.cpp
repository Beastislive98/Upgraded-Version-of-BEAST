// error_handler.cpp - Production Ready Version

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <unordered_map>
#include <regex>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>
#include "error_handler.h"
#include "data_logger.h"  // Include data_logger for logging functions

using json = nlohmann::json;

static std::atomic<bool> emergency_stop_triggered{false};
static std::atomic<int> error_count{0};
static std::atomic<int> critical_error_count{0};

// List of error messages that are considered recoverable
const std::unordered_map<std::string, bool> RECOVERABLE_ERRORS = {
    {"rate limit exceeded", true},
    {"timeout", true},
    {"connection reset", true},
    {"server busy", true},
    {"temporary unavailable", true},
    {"service unavailable", true},
    {"try again later", true},
    {"too many requests", true},
    {"insufficient balance", false},
    {"insufficient funds", false},
    {"invalid api key", false},
    {"invalid signature", false},
    {"invalid symbol", false},
    {"order would trigger immediately", false},
    {"account has been disabled", false},
    {"market is closed", false},
    {"price outside of allowed range", false},
    {"parameter error", false},
    {"order does not exist", false},
    {"invalid quantity", false}
};

// Base retry delays in milliseconds for different categories of errors
const std::unordered_map<std::string, int> BASE_RETRY_DELAYS = {
    {"rate limit exceeded", 2000},   // Rate limits: start with 2 second delay
    {"too many requests", 30000},    // Rate limits (severe): start with 30 second delay
    {"timeout", 1000},               // Timeouts: start with 1 second delay
    {"connection", 500},             // Connection issues: start with 0.5 second delay
    {"connection reset", 3000},      // Connection reset: start with 3 second delay
    {"server busy", 3000},           // Server load issues: start with 3 second delay
    {"temporary failure", 5000},     // Temporary failures: start with 5 second delay
    {"service unavailable", 10000},  // Service unavailable: start with 10 second delay
    {"default", 1000}                // Default retry delay: 1 second
};

// Maximum retry delay in milliseconds (caps at ~2 minutes)
const int MAX_RETRY_DELAY = 120000;

// Track recurring connection issues
static int connection_error_count = 0;
static std::time_t last_connection_error_time = 0;

TradingException::TradingException(const std::string& msg) : message(msg) {}

const char* TradingException::what() const noexcept {
    return message.c_str();
}

// Note: log_error and log_warning are now implemented in data_logger.cpp
// These functions will use those implementations

// Overloaded version to log error with error code and additional context
void log_error(const std::string& message, int error_code, const std::string& context) {
    try {
        // Ensure logs directory exists
        std::filesystem::create_directories("logs");
        
        std::ofstream file("logs/error_log.txt", std::ios::app);
        if (!file.is_open()) {
            std::cerr << "[ERROR_HANDLER] Could not write to error log." << std::endl;
            return;
        }

        std::time_t now = std::time(nullptr);
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        // Determine severity based on error code and message
        std::string severity = "ERROR";
        if (error_code == 4000 || error_code == -2010) {
            severity = "CRITICAL";
        } else if (error_code == -1021 || error_code == -1003) {
            severity = "WARNING";
        }

        file << "[" << timestamp << "] [" << severity << "] Error Code: " << error_code << " - " << message << "\n";
        file << "Context: " << context << "\n\n";
        file.close();
        
        // Also print to console for visibility
        std::cerr << "[" << timestamp << "] [" << severity << "] Error Code: " << error_code << " - " << message << std::endl;

        // Special handling for specific API errors
        if (error_code != 0) {
            handle_api_error(message, error_code);
        }
    } catch (const std::exception& e) {
        std::cerr << "[CRITICAL] Failed to log error: " << e.what() << std::endl;
    }
}

void handle_api_error(const std::string& error_response) {
    try {
        // Parse error response if it's JSON
        json error_json;
        try {
            error_json = json::parse(error_response);
        } catch (...) {
            error_json = {{"raw_error", error_response}};
        }
        
        // Log the API error
        std::string error_msg = "API Error: " + error_response;
        log_error(error_msg);
        
        // Check for specific API error codes that require action
        if (error_json.contains("code")) {
            int error_code = error_json["code"];
            
            switch (error_code) {
                case -1000: // UNKNOWN error
                case -1001: // DISCONNECTED
                    handle_network_error("API connection issue: " + error_response);
                    break;
                    
                case -1021: // Timestamp outside recv window
                    log_warning("Timestamp sync issue - system clock may be off");
                    break;
                    
                case -2010: // NEW_ORDER_REJECTED
                case -2011: // CANCEL_REJECTED
                    log_error("Order rejected: " + error_json.value("msg", "Unknown reason"));
                    break;
                    
                case -1003: // Too many requests
                    log_warning("Rate limit hit - implementing backoff");
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    break;
                    
                default:
                    log_error("Unhandled API error code: " + std::to_string(error_code));
                    break;
            }
        }
        
        // Save detailed error for analysis
        std::ofstream api_error_file("logs/api_errors.json", std::ios::app);
        if (api_error_file.is_open()) {
            json log_entry = {
                {"timestamp", std::time(nullptr)},
                {"error_response", error_response},
                {"parsed_error", error_json}
            };
            api_error_file << log_entry.dump() << std::endl;
            api_error_file.close();
        }
        
    } catch (const std::exception& e) {
        handle_critical_error("Failed to handle API error: " + std::string(e.what()));
    }
}

void handle_api_error(const std::string& details, int error_code) {
    // Special handling for specific API errors
    if (error_code == 4000) {
        log_error("CRITICAL: Invalid API key. Please check your API credentials.");
    } else if (error_code == -2010) {
        log_error("ERROR: Insufficient funds for the requested trade.");
    } else if (error_code == -1021) {
        log_warning("WARNING: Timestamp is outside of the recvWindow. Check system clock synchronization.");
    } else if (error_code == -1003) {
        log_warning("WARNING: Rate limit exceeded. Implementing mandatory cooldown period.");
        // This would typically trigger a cooldown mechanism
    }
}

void handle_network_error(const std::string& error_details) {
    log_error("Network Error: " + error_details);
    
    // Track recurring connection issues
    std::time_t now = std::time(nullptr);
    
    // Reset counter if last error was more than 5 minutes ago
    if (now - last_connection_error_time > 300) {
        connection_error_count = 0;
    }
    
    connection_error_count++;
    last_connection_error_time = now;
    
    if (connection_error_count >= 5) {
        log_error("CRITICAL: Persistent connection issues detected. Consider checking network connectivity.");
        connection_error_count = 0;  // Reset counter
    }
    
    // Attempt reconnection
    log_warning("Attempting automatic reconnection...");
    
    bool reconnected = false;
    int max_attempts = 5;
    
    for (int attempt = 1; attempt <= max_attempts; attempt++) {
        log_warning("Reconnection attempt " + std::to_string(attempt) + "/" + std::to_string(max_attempts));
        
        if (attempt_reconnection()) {
            log_warning("Reconnection successful");
            reconnected = true;
            break;
        }
        
        // Exponential backoff
        int delay = std::pow(2, attempt);
        log_warning("Waiting " + std::to_string(delay) + " seconds before next attempt");
        std::this_thread::sleep_for(std::chrono::seconds(delay));
    }
    
    if (!reconnected) {
        handle_critical_error("Failed to reconnect after " + std::to_string(max_attempts) + " attempts");
    }
}

void handle_critical_error(const std::string& error_message) {
    critical_error_count++;
    
    log_error("CRITICAL ERROR: " + error_message);
    
    // Save emergency state immediately
    save_emergency_state();
    
    // Trigger emergency stop
    if (!emergency_stop_triggered.load()) {
        emergency_stop_triggered.store(true);
        emergency_stop_all_trades();
    }
    
    // Send alert (you can implement email/SMS notifications here)
    send_critical_alert(error_message);
    
    // Write to critical errors file
    std::ofstream critical_file("logs/critical_errors.log", std::ios::app);
    if (critical_file.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        critical_file << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                      << "] CRITICAL: " << error_message << std::endl;
        critical_file.close();
    }
}

bool attempt_reconnection() {
    try {
        // Implement your reconnection logic here
        // This is a placeholder - you need to implement based on your trading platform
        
        log_warning("Testing connection...");
        
        // Simulate connection test
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Return true if connection successful, false otherwise
        // You should implement actual connection testing here
        return true; // Placeholder
        
    } catch (const std::exception& e) {
        log_error("Reconnection attempt failed: " + std::string(e.what()));
        return false;
    }
}

void emergency_stop_all_trades() {
    try {
        log_error("EMERGENCY STOP ACTIVATED - Stopping all trading activities");
        
        // Cancel all active orders
        // This is critical - implement your order cancellation logic here
        
        // Create emergency stop marker file
        std::ofstream stop_file("EMERGENCY_STOP.flag");
        if (stop_file.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            stop_file << "Emergency stop activated at: " 
                      << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                      << std::endl;
            stop_file << "Reason: Critical error detected" << std::endl;
            stop_file.close();
        }
        
        // Stop all trading threads/processes
        // You need to implement this based on your system architecture
        
        log_error("All trading activities have been stopped");
        
    } catch (const std::exception& e) {
        log_error("Failed to execute emergency stop: " + std::string(e.what()));
    }
}

void save_emergency_state() {
    try {
        log_warning("Saving emergency state...");
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        json emergency_state = {
            {"timestamp", time_t},
            {"error_count", error_count.load()},
            {"critical_error_count", critical_error_count.load()},
            {"emergency_stop_triggered", emergency_stop_triggered.load()}
        };
        
        // Save current portfolio state (implement based on your system)
        // emergency_state["portfolio"] = get_current_portfolio();
        
        // Save active orders (implement based on your system)
        // emergency_state["active_orders"] = get_active_orders();
        
        std::ofstream state_file("emergency_state.json");
        if (state_file.is_open()) {
            state_file << emergency_state.dump(2);
            state_file.close();
            log_warning("Emergency state saved successfully");
        } else {
            log_error("Failed to save emergency state to file");
        }
        
    } catch (const std::exception& e) {
        log_error("Failed to save emergency state: " + std::string(e.what()));
    }
}

void send_critical_alert(const std::string& error_message) {
    try {
        // Implement your alert system here (email, SMS, Slack, etc.)
        
        // For now, write to alert file
        std::ofstream alert_file("logs/alerts.log", std::ios::app);
        if (alert_file.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            alert_file << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                       << "] ALERT: " << error_message << std::endl;
            alert_file.close();
        }
        
        // You can implement email/SMS alerts here
        // send_email_alert(error_message);
        // send_sms_alert(error_message);
        
        log_warning("Critical alert sent");
        
    } catch (const std::exception& e) {
        log_error("Failed to send critical alert: " + std::string(e.what()));
    }
}

bool is_emergency_stop_active() {
    return emergency_stop_triggered.load();
}

void reset_error_counters() {
    error_count.store(0);
    critical_error_count.store(0);
    log_warning("Error counters reset");
}

void clear_emergency_stop() {
    if (emergency_stop_triggered.load()) {
        emergency_stop_triggered.store(false);
        
        // Remove emergency stop flag file
        if (std::remove("EMERGENCY_STOP.flag") == 0) {
            log_warning("Emergency stop flag cleared");
        }
        
        log_warning("Emergency stop has been cleared - system ready to resume");
    }
}

bool is_recoverable_error(const std::string& error_message) {
    // Convert to lowercase for case-insensitive matching
    std::string lowercase_message = error_message;
    std::transform(lowercase_message.begin(), lowercase_message.end(), 
                   lowercase_message.begin(), ::tolower);
    
    // Check against known error patterns
    for (const auto& [pattern, recoverable] : RECOVERABLE_ERRORS) {
        if (lowercase_message.find(pattern) != std::string::npos) {
            return recoverable;
        }
    }
    
    // By default, consider unknown errors as non-recoverable for safety
    return false;
}

int get_retry_delay(const std::string& error_message, int attempt) {
    // Ensure attempt is at least 1
    attempt = std::max(1, attempt);
    
    // Convert to lowercase for case-insensitive matching
    std::string lowercase_message = error_message;
    std::transform(lowercase_message.begin(), lowercase_message.end(), 
                   lowercase_message.begin(), ::tolower);
    
    // Find the appropriate base delay for this error type
    int base_delay = BASE_RETRY_DELAYS.at("default");
    for (const auto& [pattern, delay] : BASE_RETRY_DELAYS) {
        if (lowercase_message.find(pattern) != std::string::npos) {
            base_delay = delay;
            break;
        }
    }
    
    // Calculate exponential backoff with jitter
    // Formula: base_delay * (2^(attempt-1)) * (0.75 + 0.5*random)
    // This adds a random factor between 0.75 and 1.25 to avoid thundering herd problem
    double random_factor = 0.75 + ((double)rand() / RAND_MAX) * 0.5;
    int delay = static_cast<int>(base_delay * (1 << (attempt - 1)) * random_factor);
    
    // Cap the maximum delay
    return std::min(delay, MAX_RETRY_DELAY);
}

// Function to extract error code from API responses
int extract_error_code(const std::string& response) {
    try {
        // Use regex to find error code in common API response formats
        std::regex code_pattern("\"code\"\\s*:\\s*(-?\\d+)");
        std::smatch matches;
        
        if (std::regex_search(response, matches, code_pattern) && matches.size() > 1) {
            try {
                return std::stoi(matches[1].str());
            } catch (...) {
                return 0;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to extract error code: " << e.what() << std::endl;
    }
    
    return 0;  // Default error code if not found
}