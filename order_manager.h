#ifndef ORDER_MANAGER_H
#define ORDER_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class OrderManager {
private:
    // Private constructor for singleton
    OrderManager();
    
    // Active trades storage
    std::map<std::string, json> active_trades;
    
    // Mutex for thread-safe operations
    std::mutex trades_mutex;
    
    // Trade results by pattern ID
    std::map<int, json> trade_results;
    
    // Helper functions
    std::string generate_trade_id();
    void update_portfolio(const json& order_data);
    void save_active_trades();
    
public:
    // Delete copy constructor and assignment operator
    OrderManager(const OrderManager&) = delete;
    OrderManager& operator=(const OrderManager&) = delete;
    
    // Singleton pattern - Get singleton instance
    static OrderManager& get_instance();
    
    // Order management functions
    void add_trade(const json& trade_data);
    void update_order_status(long order_id, const std::string& status, const json& order_details = json());
    void update_order_status(const std::string& order_id, const std::string& status);
    std::vector<json> get_active_orders();
    bool cancel_order(const std::string& order_id);
    void process_filled_order(const json& order_data);
    
    // Trade management functions
    json get_trade(const std::string& trade_id);
    std::vector<json> get_all_active_trades();
    void update_trade_status_internal(const std::string& trade_id, const std::string& status, double pnl);
    
    // Trade results functions
    void store_trade_result(int pattern_id, const json& result);
    json get_result_by_pattern_id(int pattern_id);
    
    // Portfolio management
    double get_available_balance();
    void update_balance(double amount);
    
    // Risk management
    bool validate_order(const json& order_data);
    double calculate_position_size(const std::string& symbol, double risk_percent);
};

#endif // ORDER_MANAGER_H