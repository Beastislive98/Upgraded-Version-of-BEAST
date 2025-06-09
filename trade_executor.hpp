// trade_executor.hpp

#ifndef TRADE_EXECUTOR_HPP
#define TRADE_EXECUTOR_HPP

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Main order sending function
bool send_order_to_binance(const std::string& payload, const std::string& api_key, 
                         const std::string& api_secret, std::string& response);

// HMAC-SHA256 signature generation
std::string hmac_sha256(const std::string &key, const std::string &data);

// Build formatted order payload from trade signal
std::string build_order_payload(const json& trade_signal);

// Process Binance API response
void process_order_response(const std::string& response, const json& original_trade);

// Enhanced execution with strategies like TWAP and Iceberg
bool send_order_with_dynamic_execution(const std::string& payload, const json& execution_params, 
                                     const std::string& api_key, const std::string& api_secret,
                                     std::string& response);

#endif // TRADE_EXECUTOR_HPP