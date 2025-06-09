#ifndef TRADE_EXECUTOR_H
#define TRADE_EXECUTOR_H

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Callback function for CURL response
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp);

// Main trading functions
std::string send_http_request(const std::string& url, const std::string& data, const std::string& headers);
bool place_order(const json& order_data);
bool cancel_order(const std::string& order_id);
json get_account_info();
json get_market_data(const std::string& symbol);

// Order processing functions
void process_order_response(const std::string& response, const json& original_order);
bool validate_order_response(const json& response);

// Utility functions
std::string generate_signature(const std::string& data, const std::string& secret);
long long get_current_timestamp();

#endif // TRADE_EXECUTOR_H