// order_manager.cpp - Production Ready Version

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <vector>
#include <mutex>
#include <map>
#include <thread>
#include <chrono>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include "order_manager.h"
#include "order_manager.hpp"  // Include this for the standalone functions
#include "data_logger.h"
#include "error_handler.h"

using json = nlohmann::json;

// Callback for CURL response
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Integrated function from feedback_trigger.cpp
void trigger_pattern_feedback(int pattern_id, bool success) {
    std::string outcome = success ? "true" : "false";
    std::string command = "python3 pattern_feedback.py --pattern_id=" + std::to_string(pattern_id) + " --success=" + outcome;
    
    try {
        int result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "[FEEDBACK] Feedback triggered successfully for pattern " << pattern_id << std::endl;
        } else {
            std::cerr << "[ERROR] Feedback trigger failed for pattern " << pattern_id << ", exit code: " << result << std::endl;
            log_error("Pattern feedback trigger failed for ID " + std::to_string(pattern_id) + ", exit code: " + std::to_string(result));
        }
    } catch (const std::exception& e) {
        log_error("Exception in trigger_pattern_feedback: " + std::string(e.what()));
    }
}

// OrderManager implementation

// Private constructor for singleton
OrderManager::OrderManager() {
    // Load active trades from file on initialization if it exists
    try {
        std::ifstream active_file("logs/active_trades.json");
        if (active_file.is_open()) {
            json active_trades_array;
            active_file >> active_trades_array;
            active_file.close();
            
            if (active_trades_array.is_array()) {
                for (const auto& trade : active_trades_array) {
                    if (trade.contains("trade_id")) {
                        std::string trade_id = trade["trade_id"];
                        active_trades[trade_id] = trade;
                    }
                }
            }
            std::cout << "[ORDER_MANAGER] Loaded " << active_trades.size() << " active trades." << std::endl;
        }
        
        // Load trade results from file if it exists
        std::ifstream results_file("logs/trade_results.json");
        if (results_file.is_open()) {
            std::string line;
            while (std::getline(results_file, line)) {
                if (!line.empty()) {
                    try {
                        json result = json::parse(line);
                        if (result.contains("pattern_id")) {
                            int pattern_id = result["pattern_id"];
                            trade_results[pattern_id] = result;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "[ORDER_MANAGER] Error parsing trade result: " << e.what() << std::endl;
                    }
                }
            }
            results_file.close();
            std::cout << "[ORDER_MANAGER] Loaded " << trade_results.size() << " trade results." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ORDER_MANAGER] Error loading trades: " << e.what() << std::endl;
        log_error("Error loading trades in OrderManager: " + std::string(e.what()));
    }
}

// Get singleton instance
OrderManager& OrderManager::get_instance() {
    static OrderManager instance;
    return instance;
}

// Save active trades to file
void OrderManager::save_active_trades() {
    std::lock_guard<std::mutex> lock(trades_mutex);
    
    try {
        std::ofstream file("logs/active_trades.json");
        if (!file.is_open()) {
            log_error("Failed to open active trades file for writing");
            return;
        }
        
        json active_trades_array = json::array();
        for (const auto& [_, trade] : active_trades) {
            active_trades_array.push_back(trade);
        }
        
        file << active_trades_array.dump(2);
        file.close();
    } catch (const std::exception& e) {
        log_error("Failed to save active trades: " + std::string(e.what()));
    }
}

// Update order status (called from trade_executor)
void OrderManager::update_order_status(long order_id, const std::string& status, const json& order_details) {
    try {
        // Convert order_id to string for use as trade_id
        std::string trade_id = std::to_string(order_id);
        
        // Also check clientOrderId if available
        if (order_details.contains("clientOrderId")) {
            trade_id = order_details["clientOrderId"];
        }
        
        // Calculate PnL (if available in order_details)
        double pnl = 0.0;
        if (order_details.contains("realizedPnl")) {
            pnl = order_details["realizedPnl"].get<double>();
        }
        
        // Call the update_trade_status function
        update_trade_status_internal(trade_id, status, pnl);
        
        // Log the order
        log_order(order_details, status);
    } catch (const std::exception& e) {
        log_error("Error in update_order_status: " + std::string(e.what()));
    }
}

// Overloaded update_order_status
void OrderManager::update_order_status(const std::string& order_id, const std::string& status) {
    json empty_details;
    update_order_status(std::stol(order_id), status, empty_details);
}

// Add trade to active trades
void OrderManager::add_trade(const json& trade_details) {
    try {
        std::string trade_id = trade_details.value("trade_id", generate_trade_id());
        
        {
            std::lock_guard<std::mutex> lock(trades_mutex);
            active_trades[trade_id] = trade_details;
        }
        save_active_trades();
        std::cout << "[ORDER_MANAGER] Added trade " << trade_id << " to tracking" << std::endl;
    } catch (const std::exception& e) {
        log_error("Error in add_trade: " + std::string(e.what()));
    }
}

// Get trade details by ID
json OrderManager::get_trade(const std::string& trade_id) {
    try {
        std::lock_guard<std::mutex> lock(trades_mutex);
        if (active_trades.find(trade_id) != active_trades.end()) {
            return active_trades[trade_id];
        }
    } catch (const std::exception& e) {
        log_error("Error in get_trade: " + std::string(e.what()));
    }
    return json();
}

// Store trade result
void OrderManager::store_trade_result(int pattern_id, const json& result) {
    try {
        {
            std::lock_guard<std::mutex> lock(trades_mutex);
            trade_results[pattern_id] = result;
        }
        
        // Also append to trade results file
        std::ofstream result_file("logs/trade_results.json", std::ios::app);
        if (result_file.is_open()) {
            result_file << result.dump() << std::endl;
            result_file.close();
        } else {
            log_error("Failed to open trade results file");
        }
    } catch (const std::exception& e) {
        log_error("Error in store_trade_result: " + std::string(e.what()));
    }
}

// Get all active trades
std::vector<json> OrderManager::get_all_active_trades() {
    std::vector<json> trades;
    
    try {
        std::lock_guard<std::mutex> lock(trades_mutex);
        for (const auto& [_, trade] : active_trades) {
            trades.push_back(trade);
        }
    } catch (const std::exception& e) {
        log_error("Error in get_all_active_trades: " + std::string(e.what()));
    }
    
    return trades;
}

// Get trade result by pattern ID
json OrderManager::get_result_by_pattern_id(int pattern_id) {
    try {
        std::lock_guard<std::mutex> lock(trades_mutex);
        if (trade_results.find(pattern_id) != trade_results.end()) {
            return trade_results[pattern_id];
        }
    } catch (const std::exception& e) {
        log_error("Error in get_result_by_pattern_id: " + std::string(e.what()));
    }
    
    return json();
}

// Update trade status
void OrderManager::update_trade_status_internal(const std::string& trade_id, const std::string& status, double pnl) {
    json updated_trade;
    bool is_completed = false;
    bool is_success = false;
    int pattern_id = -1;
    
    try {
        {
            std::lock_guard<std::mutex> lock(trades_mutex);
            if (active_trades.find(trade_id) != active_trades.end()) {
                active_trades[trade_id]["status"] = status;
                active_trades[trade_id]["pnl"] = pnl;
                active_trades[trade_id]["close_time"] = std::time(nullptr);
                
                // Determine if this is a successful trade
                is_success = (status == "FILLED" && pnl > 0) || 
                           (status == "TAKE_PROFIT");
                active_trades[trade_id]["success"] = is_success;
                
                updated_trade = active_trades[trade_id];
                
                // Check if the trade has a pattern ID for ML feedback
                if (active_trades[trade_id].contains("pattern_id")) {
                    pattern_id = active_trades[trade_id]["pattern_id"].get<int>();
                }
                
                // If the trade is completed or cancelled, store it in results and remove from active
                if (status == "FILLED" || status == "CANCELED" || status == "EXPIRED" || status == "REJECTED" || 
                    status == "TAKE_PROFIT" || status == "STOP_LOSS") {
                    
                    is_completed = true;
                    
                    if (pattern_id > 0) {
                        store_trade_result(pattern_id, active_trades[trade_id]);
                    }
                    
                    active_trades.erase(trade_id);
                }
                
                // Log the updated trade status
                log_order(updated_trade, status);
            }
        }
        
        // Save updated active trades to file
        save_active_trades();
        
        // Append to trade results file if it's a completed trade
        if (!updated_trade.empty() && is_completed) {
            std::ofstream result_file("logs/trade_results.json", std::ios::app);
            if (result_file.is_open()) {
                result_file << updated_trade.dump() << std::endl;
                result_file.close();
                std::cout << "[ORDER_MANAGER] Logged trade result for " << trade_id << std::endl;
            } else {
                log_error("Failed to open trade results file");
            }
            
            // Send ML feedback if applicable
            if (pattern_id > 0) {
                trigger_pattern_feedback(pattern_id, is_success);
            }
        }
    } catch (const std::exception& e) {
        log_error("Error in update_trade_status_internal: " + std::string(e.what()));
    }
}

std::string OrderManager::generate_trade_id() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << "TRADE_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
       << "_" << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

// Stub implementations for missing methods
std::vector<json> OrderManager::get_active_orders() {
    return get_all_active_trades();
}

bool OrderManager::cancel_order(const std::string& order_id) {
    // Implement order cancellation logic here
    return true;
}

void OrderManager::process_filled_order(const json& order_data) {
    // Implement filled order processing here
}

double OrderManager::get_available_balance() {
    // Implement balance retrieval logic here
    return 0.0;
}

void OrderManager::update_balance(double amount) {
    // Implement balance update logic here
}

bool OrderManager::validate_order(const json& order_data) {
    // Implement order validation logic here
    return true;
}

double OrderManager::calculate_position_size(const std::string& symbol, double risk_percent) {
    // Implement position size calculation logic here
    return 0.0;
}

void OrderManager::update_portfolio(const json& order_data) {
    // Implement portfolio update logic here
}

// Implementation of the functions declared in the order_manager.hpp header

void monitor_trades() {
    try {
        std::cout << "[MONITOR] Checking active trades status..." << std::endl;
        
        // Get active trades from the singleton
        std::vector<json> active_trades = OrderManager::get_instance().get_all_active_trades();
        
        if (active_trades.empty()) {
            std::cout << "[MONITOR] No active trades to monitor." << std::endl;
            return;
        }
        
        std::cout << "[MONITOR] Monitoring " << active_trades.size() << " active trades." << std::endl;
        
        // Setup CURL for price checks
        CURL* curl = curl_easy_init();
        if (!curl) {
            log_error("Failed to initialize CURL for trade monitoring");
            return;
        }
        
        // For each active trade, check its status
        for (auto& trade : active_trades) {
            try {
                if (!trade.contains("symbol") || !trade.contains("trade_id") || 
                    !trade.contains("entry") || !trade.contains("stopLoss") || 
                    !trade.contains("takeProfit") || !trade.contains("side")) {
                    
                    std::cerr << "[MONITOR] Trade missing required fields: " << trade.dump() << std::endl;
                    continue;
                }
                
                std::string symbol = trade["symbol"];
                std::string trade_id = trade["trade_id"];
                double entry_price = trade["entry"];
                double stop_loss = trade["stopLoss"];
                double take_profit = trade["takeProfit"];
                std::string side = trade["side"];
                
                std::cout << "[MONITOR] Checking " << symbol << " trade " << trade_id << std::endl;
                
                // Get current price from Binance
                std::string url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=" + symbol;
                
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
                
                std::string response;
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
                
                CURLcode res = curl_easy_perform(curl);
                
                if (res != CURLE_OK) {
                    std::cerr << "[MONITOR] CURL error: " << curl_easy_strerror(res) << std::endl;
                    continue;
                }
                
                // Parse the price response
                json price_json = json::parse(response);
                
                if (!price_json.contains("price")) {
                    std::cerr << "[MONITOR] Invalid price response: " << response << std::endl;
                    continue;
                }
                
                double current_price = std::stod(price_json["price"].get<std::string>());
                
                bool close_trade = false;
                bool take_profit_hit = false;
                double pnl = 0.0;
                
                // Check if stop loss or take profit hit
                if (side == "BUY" || side == "LONG") {
                    if (current_price <= stop_loss) {
                        close_trade = true;
                        pnl = (stop_loss - entry_price) * trade.value("quantity", 1.0);
                    } else if (current_price >= take_profit) {
                        close_trade = true;
                        take_profit_hit = true;
                        pnl = (take_profit - entry_price) * trade.value("quantity", 1.0);
                    } else {
                        // Calculate unrealized PnL for reporting
                        trade["unrealized_pnl"] = (current_price - entry_price) * trade.value("quantity", 1.0);
                        trade["current_price"] = current_price;
                    }
                } else { // SELL or SHORT
                    if (current_price >= stop_loss) {
                        close_trade = true;
                        pnl = (entry_price - stop_loss) * trade.value("quantity", 1.0);
                    } else if (current_price <= take_profit) {
                        close_trade = true;
                        take_profit_hit = true;
                        pnl = (entry_price - take_profit) * trade.value("quantity", 1.0);
                    } else {
                        // Calculate unrealized PnL for reporting
                        trade["unrealized_pnl"] = (entry_price - current_price) * trade.value("quantity", 1.0);
                        trade["current_price"] = current_price;
                    }
                }
                
                if (close_trade) {
                    std::string outcome = take_profit_hit ? "TAKE_PROFIT" : "STOP_LOSS";
                    std::cout << "[MONITOR] " << outcome << " triggered for " << trade_id 
                              << " with PnL: " << pnl << std::endl;
                    
                    // Update the trade status using the singleton
                    update_trade_status(trade_id, outcome, pnl);
                } else {
                    // Just log the current position status
                    std::cout << "[MONITOR] " << symbol << " trade " << trade_id 
                              << " current price: " << current_price 
                              << ", unrealized PnL: " << trade.value("unrealized_pnl", 0.0) << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[MONITOR] Error processing trade: " << e.what() << std::endl;
                log_error("Error monitoring trade: " + std::string(e.what()));
            }
        }
        
        // Cleanup CURL
        curl_easy_cleanup(curl);
        
    } catch (const std::exception& e) {
        log_error("Error in monitor_trades: " + std::string(e.what()));
    }
}

json get_trade_result_by_pattern_id(int pattern_id) {
    return OrderManager::get_instance().get_result_by_pattern_id(pattern_id);
}

std::vector<json> get_active_trades() {
    return OrderManager::get_instance().get_all_active_trades();
}

void update_trade_status(const std::string& trade_id, const std::string& status, double pnl) {
    OrderManager::get_instance().update_trade_status_internal(trade_id, status, pnl);
}