// error_handler.hpp

#ifndef ERROR_HANDLER_HPP
#define ERROR_HANDLER_HPP

#include <string>

// Log error with message only
void log_error(const std::string& message);

// Log error with error code and context
void log_error(const std::string& message, int error_code, const std::string& context);

// Check if an error is potentially recoverable
bool is_recoverable_error(const std::string& error_message);

// Calculate retry delay with exponential backoff
int get_retry_delay(const std::string& error_message, int attempt);

// Handle specific types of errors
void handle_connection_error(const std::string& details);
void handle_api_error(const std::string& details, int error_code);
void handle_json_error(const std::string& details);

// Extract error code from API responses
int extract_error_code(const std::string& response);

#endif // ERROR_HANDLER_HPP