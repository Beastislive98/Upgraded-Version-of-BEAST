// order_manager.hpp

#ifndef ORDER_MANAGER_HPP
#define ORDER_MANAGER_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Log order details with status
void log_order(const json& order, const std::string& status);

// Monitor active trades for SL/TP hits
void monitor_trades();

// Get trade result by pattern ID
json get_trade_result_by_pattern_id(int pattern_id);

// Get all active trades
std::vector<json> get_active_trades();

// Update trade status
void update_trade_status(const std::string& trade_id, const std::string& status, double pnl);

// Trigger pattern feedback to ML system
void trigger_pattern_feedback(int pattern_id, bool success);

#endif // ORDER_MANAGER_HPP