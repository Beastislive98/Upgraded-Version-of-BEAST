#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <string>
#include <exception>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class TradingException : public std::exception {
private:
    std::string message;
    
public:
    explicit TradingException(const std::string& msg);
    const char* what() const noexcept override;
};

// Error handling functions
void log_error(const std::string& error_message);
void log_error(const std::string& message, int error_code, const std::string& context);
void log_warning(const std::string& warning_message);
void handle_api_error(const std::string& error_response);
void handle_api_error(const std::string& details, int error_code);
void handle_network_error(const std::string& error_details);
void handle_critical_error(const std::string& error_message);

// Error recovery functions
bool attempt_reconnection();
void emergency_stop_all_trades();
void save_emergency_state();

// Additional production functions
void send_critical_alert(const std::string& error_message);
bool is_emergency_stop_active();
void reset_error_counters();
void clear_emergency_stop();

// Error analysis functions
bool is_recoverable_error(const std::string& error_message);
int get_retry_delay(const std::string& error_message, int attempt);
int extract_error_code(const std::string& response);

#endif // ERROR_HANDLER_H