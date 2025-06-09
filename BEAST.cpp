// BEAST.cpp - Production Ready Version with TP/SL Order Placement

#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <openssl/hmac.h>
#include <iomanip>
#include <sstream>
#include "trade_executor.hpp"
#include "order_manager.hpp"
#include "error_handler.hpp"
#include "data_logger.hpp"
#include "order_manager.h"

using json = nlohmann::json;

// Forward declarations
void add_active_trade(const json& order);
bool place_stop_loss_order(const std::string& symbol, const std::string& original_side, 
                           const std::string& positionSide, double quantity, double stopLoss,
                           const std::string& api_key, const std::string& api_secret);
bool place_take_profit_order(const std::string& symbol, const std::string& original_side,
                            const std::string& positionSide, double quantity, double takeProfit,
                            const std::string& api_key, const std::string& api_secret);

// Symbol precision map - common Binance Futures precision settings (FALLBACK)
static const std::unordered_map<std::string, int> SYMBOL_PRECISION = {
    // Major pairs - usually 3 decimal places
    {"BTCUSDT", 3}, {"ETHUSDT", 3}, {"BNBUSDT", 2}, {"ADAUSDT", 0},
    {"XRPUSDT", 0}, {"SOLUSDT", 0}, {"DOTUSDT", 1}, {"DOGEUSDT", 0},
    {"AVAXUSDT", 1}, {"SHIBUSDT", 0}, {"LINKUSDT", 1}, {"MATICUSDT", 0},
    {"UNIUSDT", 1}, {"LTCUSDT", 2}, {"BCHUSDT", 3}, {"ALGOUSDT", 0},
    
    // Alt coins - usually 0-2 decimal places
    {"ARPAUSDT", 0}, {"BAKEUSDT", 0}, {"BBUSDT", 0}, {"AAVEUSDT", 2},
    {"COMPUSDT", 3}, {"MKRUSDT", 4}, {"YFIUSDT", 5}, {"SUSHIUSDT", 0},
    {"CRVUSDT", 0}, {"1INCHUSDT", 0}, {"ALPHAUSDT", 0}, {"ATMUSDT", 1},
    {"AXSUSDT", 0}, {"BANDUSDT", 1}, {"BATUSDT", 0}, {"CELRUSDT", 0},
    {"CHZUSDT", 0}, {"COTIUSDT", 0}, {"CTKUSDT", 0}, {"DASHUSDT", 3},
    {"EGGSUSDT", 0}, {"ENJUSDT", 0}, {"ETCUSDT", 2}, {"FETUSDT", 0},
    {"FILUSDT", 2}, {"FLMUSDT", 0}, {"FTMUSDT", 0}, {"GALAUSDT", 0},
    {"GMTUSDT", 0}, {"GRTUSDT", 0}, {"HBARUSDT", 0}, {"ICPUSDT", 2},
    {"IOSTUSDT", 0}, {"IOTXUSDT", 0}, {"JASMYUSDT", 0}, {"KAVAUSDT", 1},
    {"KLAYUSDT", 0}, {"KNCUSDT", 1}, {"LDOUSDT", 1}, {"MANAUSDT", 0},
    {"NEARUSDT", 1}, {"OCEANUSDT", 0}, {"ONTUSDT", 0}, {"QTUMUSDT", 1},
    {"RAYUSDT", 1}, {"ROSEUSDT", 0}, {"RUNEUSDT", 1}, {"SANDUSDT", 0},
    {"SKLUSDT", 0}, {"SNXUSDT", 1}, {"STORJUSDT", 1}, {"THETAUSDT", 0},
    {"TLMUSDT", 0}, {"TOMOUSDT", 1}, {"TRXUSDT", 0}, {"VETUSDT", 0},
    {"WAVESUSDT", 1}, {"WOOUSDT", 0}, {"XLMUSDT", 0}, {"XMRUSDT", 3},
    {"XTZUSDT", 1}, {"ZECUSDT", 3}, {"ZENUSDT", 1}, {"ZILUSDT", 0},
    {"SCRUSDT", 0}, {"KERNELUSDT", 0}  // Added based on your error logs
};

// Dynamic precision maps - populated from Binance API
static std::unordered_map<std::string, int> DYNAMIC_SYMBOL_PRECISION;
static std::unordered_map<std::string, double> DYNAMIC_MIN_NOTIONAL;
static std::unordered_map<std::string, double> DYNAMIC_MIN_QUANTITY;
static bool dynamic_precision_loaded = false;

// Callback function for CURL to capture response (MOVED HERE)
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Function to fetch all symbol precision rules from Binance
bool initialize_symbol_precision_from_binance(const std::string& api_key) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[PRECISION] Failed to initialize CURL for exchange info" << std::endl;
        return false;
    }
    
    try {
        std::string url = "https://fapi.binance.com/fapi/v1/exchangeInfo";
        std::string response;
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("X-MBX-APIKEY: " + api_key).c_str());
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        
        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
        
        if (res != CURLE_OK) {
            std::cerr << "[PRECISION] CURL error fetching exchange info: " << curl_easy_strerror(res) << std::endl;
            return false;
        }
        
        if (response.empty()) {
            std::cerr << "[PRECISION] Empty response from exchange info" << std::endl;
            return false;
        }
        
        // Parse the JSON response
        json exchange_info = json::parse(response);
        
        if (!exchange_info.contains("symbols") || !exchange_info["symbols"].is_array()) {
            std::cerr << "[PRECISION] Invalid exchange info format" << std::endl;
            return false;
        }
        
        int symbols_processed = 0;
        
        // Process each symbol
        for (const auto& symbol_info : exchange_info["symbols"]) {
            if (!symbol_info.contains("symbol") || !symbol_info.contains("quantityPrecision")) {
                continue;
            }
            
            std::string symbol = symbol_info["symbol"];
            int quantity_precision = symbol_info["quantityPrecision"];
            
            // Store quantity precision
            DYNAMIC_SYMBOL_PRECISION[symbol] = quantity_precision;
            
            // Default values
            DYNAMIC_MIN_NOTIONAL[symbol] = 5.0;  // Default $5 minimum
            DYNAMIC_MIN_QUANTITY[symbol] = std::pow(10.0, -quantity_precision);
            
            // Extract filters for more precise rules
            if (symbol_info.contains("filters") && symbol_info["filters"].is_array()) {
                for (const auto& filter : symbol_info["filters"]) {
                    if (!filter.contains("filterType")) continue;
                    
                    std::string filter_type = filter["filterType"];
                    
                    // MIN_NOTIONAL filter
                    if (filter_type == "MIN_NOTIONAL" && filter.contains("notional")) {
                        try {
                            double notional = std::stod(filter["notional"].get<std::string>());
                            DYNAMIC_MIN_NOTIONAL[symbol] = notional;
                        } catch (...) {
                            // Keep default if parsing fails
                        }
                    }
                    
                    // LOT_SIZE filter
                    else if (filter_type == "LOT_SIZE" && filter.contains("minQty")) {
                        try {
                            double min_qty = std::stod(filter["minQty"].get<std::string>());
                            DYNAMIC_MIN_QUANTITY[symbol] = min_qty;
                        } catch (...) {
                            // Keep default if parsing fails
                        }
                    }
                }
            }
            
            symbols_processed++;
        }
        
        dynamic_precision_loaded = true;
        std::cout << "[PRECISION] ✅ Successfully loaded precision rules for " 
                  << symbols_processed << " symbols from Binance" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[PRECISION] Error fetching exchange info: " << e.what() << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }
}

// Function to get symbol precision (enhanced with dynamic lookup)
int get_symbol_precision(const std::string& symbol) {
    // First try dynamic precision if available
    if (dynamic_precision_loaded) {
        auto it = DYNAMIC_SYMBOL_PRECISION.find(symbol);
        if (it != DYNAMIC_SYMBOL_PRECISION.end()) {
            return it->second;
        }
    }
    
    // Fallback to hardcoded precision map
    auto it = SYMBOL_PRECISION.find(symbol);
    return (it != SYMBOL_PRECISION.end()) ? it->second : 0;
}

// Function to get minimum notional value for symbol
double get_min_notional(const std::string& symbol) {
    if (dynamic_precision_loaded) {
        auto it = DYNAMIC_MIN_NOTIONAL.find(symbol);
        if (it != DYNAMIC_MIN_NOTIONAL.end()) {
            return it->second;
        }
    }
    return 5.0;  // Default $5 minimum
}

// Function to get minimum quantity for symbol
double get_min_quantity(const std::string& symbol) {
    if (dynamic_precision_loaded) {
        auto it = DYNAMIC_MIN_QUANTITY.find(symbol);
        if (it != DYNAMIC_MIN_QUANTITY.end()) {
            return it->second;
        }
    }
    
    // Fallback calculation
    int precision = get_symbol_precision(symbol);
    return (precision > 0) ? std::pow(10.0, -precision) : 1.0;
}

// Function to round quantity to correct precision
double round_to_precision(double value, int precision) {
    if (precision <= 0) {
        return std::floor(value);  // Round down to whole number
    }
    
    double multiplier = std::pow(10.0, precision);
    return std::floor(value * multiplier) / multiplier;
}

// Function to validate and fix quantity precision
double fix_quantity_precision(const std::string& symbol, double quantity, const std::string& api_key) {
    // Get precision from dynamic map (or fallback to hardcoded)
    int precision = get_symbol_precision(symbol);
    
    double rounded_quantity = round_to_precision(quantity, precision);
    
    // Ensure minimum quantity using dynamic rules
    double min_quantity = get_min_quantity(symbol);
    if (rounded_quantity < min_quantity) {
        rounded_quantity = min_quantity;
    }
    
    std::cout << "[PRECISION] " << symbol << " quantity " << quantity 
              << " rounded to " << rounded_quantity 
              << " (precision: " << precision << ", min: " << min_quantity << ")" << std::endl;
    
    return rounded_quantity;
}

// Function to load API keys from .env file
bool load_env_file(std::string& api_key, std::string& api_secret) {
    std::ifstream env_file(".env");
    if (!env_file.is_open()) {
        std::cout << "[CONFIG] .env file not found, trying environment variables..." << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(env_file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Find the = sign
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // Remove quotes if present
        if (value.size() >= 2 && value[0] == '"' && value[value.size()-1] == '"') {
            value = value.substr(1, value.size()-2);
        }
        
        if (key == "BINANCE_API_KEY") {
            api_key = value;
        } else if (key == "BINANCE_API_SECRET") {
            api_secret = value;
        }
    }
    
    env_file.close();
    
    if (!api_key.empty() && !api_secret.empty()) {
        std::cout << "[CONFIG] API keys loaded from .env file successfully" << std::endl;
        return true;
    }
    
    return false;
}

// Function to get API credentials from .env file or environment variables
bool get_api_credentials(std::string& api_key, std::string& api_secret) {
    // First try to load from .env file
    if (load_env_file(api_key, api_secret)) {
        return true;
    }
    
    // Fallback to environment variables
    char* api_key_env = std::getenv("BINANCE_API_KEY");
    char* api_secret_env = std::getenv("BINANCE_API_SECRET");
    
    api_key = api_key_env ? api_key_env : "";
    api_secret = api_secret_env ? api_secret_env : "";
    
    if (!api_key.empty() && !api_secret.empty()) {
        std::cout << "[CONFIG] API keys loaded from environment variables" << std::endl;
        return true;
    }
    
    return false;
}

// Function to fetch current price for a symbol from Binance
double fetch_current_price(const std::string& symbol, const std::string& api_key) {
    CURL* curl = curl_easy_init();
    if (!curl) return 0.0;
    
    try {
        std::string url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=" + symbol;
        std::string response;
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("X-MBX-APIKEY: " + api_key).c_str());
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        
        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
        
        if (res == CURLE_OK && !response.empty()) {
            try {
                json price_data = json::parse(response);
                if (price_data.contains("price")) {
                    double price = std::stod(price_data["price"].get<std::string>());
                    std::cout << "[PRICE] " << symbol << " current price: $" << price << std::endl;
                    return price;
                }
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Failed to parse price response: " << e.what() << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to fetch price for " << symbol << ": " << e.what() << std::endl;
    }
    
    return 0.0;
}

// Function to ensure minimum notional value (enhanced with dynamic rules)
double ensure_minimum_notional(const std::string& symbol, double quantity, const std::string& api_key) {
    // Get minimum notional from dynamic rules
    double MIN_NOTIONAL = get_min_notional(symbol);
    
    // Fetch current price
    double current_price = fetch_current_price(symbol, api_key);
    if (current_price <= 0.0) {
        std::cerr << "[ERROR] Could not fetch price for " << symbol << ", using original quantity" << std::endl;
        return quantity;
    }
    
    // Calculate current notional value
    double current_notional = quantity * current_price;
    
    std::cout << "[NOTIONAL] " << symbol << " - Quantity: " << quantity 
              << ", Price: $" << current_price 
              << ", Value: $" << current_notional 
              << ", Min Required: $" << MIN_NOTIONAL << std::endl;
    
    // Check if notional value is sufficient
    if (current_notional >= MIN_NOTIONAL) {
        std::cout << "[NOTIONAL] ✅ Order value ($" << current_notional << ") meets minimum requirement" << std::endl;
        return quantity;
    }
    
    // Calculate minimum quantity needed
    double min_quantity_needed = MIN_NOTIONAL / current_price;
    
    // Apply precision rounding to the minimum quantity
    int precision = get_symbol_precision(symbol);
    double adjusted_quantity = round_to_precision(min_quantity_needed, precision);
    
    // Ensure we don't go below the calculated minimum
    if (adjusted_quantity * current_price < MIN_NOTIONAL) {
        // Add one more unit if rounding down caused us to go below minimum
        double increment = (precision > 0) ? std::pow(10.0, -precision) : 1.0;
        adjusted_quantity += increment;
        
        // Re-round after adding increment to maintain precision
        adjusted_quantity = round_to_precision(adjusted_quantity, precision);
    }
    
    double final_notional = adjusted_quantity * current_price;
    
    std::cout << "[NOTIONAL] ⚠️ Original value ($" << current_notional 
              << ") below minimum. Adjusted quantity: " << quantity 
              << " → " << adjusted_quantity 
              << " (Final value: $" << final_notional << ")" << std::endl;
    
    return adjusted_quantity;
}

// NEW FUNCTION: Place Stop Loss Order
bool place_stop_loss_order(const std::string& symbol, const std::string& original_side, 
                           const std::string& positionSide, double quantity, double stopLoss,
                           const std::string& api_key, const std::string& api_secret) {
    try {
        // Determine the correct side for SL order (opposite of entry)
        std::string sl_side = (original_side == "BUY" || original_side == "LONG") ? "SELL" : "BUY";
        
        json sl_payload = {
            {"symbol", symbol},
            {"side", sl_side},
            {"type", "STOP_MARKET"},
            {"positionSide", positionSide},
            {"quantity", quantity},
            {"stopPrice", stopLoss},
            {"reduceOnly", true},  // Important: SL should reduce position
            {"newOrderRespType", "RESULT"}
        };

        std::string sl_client_order_id = "beast_sl_" + std::to_string(std::time(nullptr)) + "_" + 
                                       std::to_string(std::rand() % 10000);
        sl_payload["newClientOrderId"] = sl_client_order_id;
        
        log_order(sl_payload, "SENDING_SL");
        
        std::string sl_response;
        bool success = send_order_to_binance(sl_payload.dump(), api_key, api_secret, sl_response);
        
        if (success) {
            json sl_response_json = json::parse(sl_response);
            if (sl_response_json.contains("code") && sl_response_json["code"] != 0) {
                log_error("Stop Loss order failed: " + sl_response_json.value("msg", "Unknown error"));
                return false;
            }
            log_order(sl_response_json, "SL_EXECUTED");
            std::cout << "[BEAST] ✅ Stop Loss order placed at: $" << stopLoss << std::endl;
        } else {
            log_error("Failed to send Stop Loss order to Binance");
        }
        
        return success;
        
    } catch (const std::exception& e) {
        log_error("Exception in place_stop_loss_order: " + std::string(e.what()));
        return false;
    }
}

// NEW FUNCTION: Place Take Profit Order
bool place_take_profit_order(const std::string& symbol, const std::string& original_side,
                            const std::string& positionSide, double quantity, double takeProfit,
                            const std::string& api_key, const std::string& api_secret) {
    try {
        // Determine the correct side for TP order (opposite of entry)
        std::string tp_side = (original_side == "BUY" || original_side == "LONG") ? "SELL" : "BUY";
        
        json tp_payload = {
            {"symbol", symbol},
            {"side", tp_side},
            {"type", "TAKE_PROFIT_MARKET"},
            {"positionSide", positionSide},
            {"quantity", quantity},
            {"stopPrice", takeProfit},
            {"reduceOnly", true},  // Important: TP should reduce position
            {"newOrderRespType", "RESULT"}
        };

        std::string tp_client_order_id = "beast_tp_" + std::to_string(std::time(nullptr)) + "_" + 
                                       std::to_string(std::rand() % 10000);
        tp_payload["newClientOrderId"] = tp_client_order_id;
        
        log_order(tp_payload, "SENDING_TP");
        
        std::string tp_response;
        bool success = send_order_to_binance(tp_payload.dump(), api_key, api_secret, tp_response);
        
        if (success) {
            json tp_response_json = json::parse(tp_response);
            if (tp_response_json.contains("code") && tp_response_json["code"] != 0) {
                log_error("Take Profit order failed: " + tp_response_json.value("msg", "Unknown error"));
                return false;
            }
            log_order(tp_response_json, "TP_EXECUTED");
            std::cout << "[BEAST] ✅ Take Profit order placed at: $" << takeProfit << std::endl;
        } else {
            log_error("Failed to send Take Profit order to Binance");
        }
        
        return success;
        
    } catch (const std::exception& e) {
        log_error("Exception in place_take_profit_order: " + std::string(e.what()));
        return false;
    }
}

// UPDATED FUNCTION: Enhanced load_and_send_trade with TP/SL order placement
bool load_and_send_trade(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            log_error("Could not open JSON signal file: " + file_path);
            return false;
        }

        json trade;
        file >> trade;
        file.close();
        
        // Validate the trade signal has required fields
        std::vector<std::string> required_fields = {"symbol", "side"};
        for (const auto& field : required_fields) {
            if (!trade.contains(field) || trade[field].empty()) {
                log_error("Trade signal missing required field: " + field);
                return false;
            }
        }

        // Extract required Binance Futures fields
        std::string symbol = trade.value("symbol", "");
        std::string side = trade.value("side", "BUY");
        std::string type = trade.value("type", "MARKET");
        std::string positionSide = trade.value("positionSide", "LONG");
        std::string marginType = trade.value("marginType", "ISOLATED");
        bool reduceOnly = trade.value("reduceOnly", false);

        double raw_quantity = trade.value("quantity", 0.0);
        double price = trade.value("price", 0.0);
        double stopPrice = trade.value("stopPrice", 0.0);
        
        // Extract TP/SL values from signal
        double stopLoss = trade.value("stopLoss", 0.0);
        double takeProfit = trade.value("takeProfit", 0.0);
        
        // Verify minimum quantity
        if (raw_quantity <= 0.0) {
            log_error("Invalid quantity in trade signal: " + std::to_string(raw_quantity));
            return false;
        }

        // Get API credentials first (needed for price fetching)
        std::string api_key, api_secret;
        if (!get_api_credentials(api_key, api_secret)) {
            log_error("API credentials missing. Please check .env file or environment variables");
            return false;
        }

        // Step 1: Fix quantity precision based on symbol requirements
        double precision_fixed_quantity = fix_quantity_precision(symbol, raw_quantity, api_key);
        
        // Step 2: Ensure minimum notional value
        double quantity = ensure_minimum_notional(symbol, precision_fixed_quantity, api_key);
        
        // Final validation of adjusted quantity
        if (quantity <= 0.0) {
            log_error("Quantity became zero after adjustments for " + symbol);
            return false;
        }

        // Build ENTRY order payload
        json payload = {
            {"symbol", symbol},
            {"side", side},
            {"type", type},
            {"positionSide", positionSide},
            {"quantity", quantity},  // Using precision-fixed and notional-adjusted quantity
            {"newOrderRespType", "RESULT"}
        };

        // Only add timeInForce for order types that require it (NOT for MARKET orders)
        if (type != "MARKET") {
            payload["timeInForce"] = trade.value("timeInForce", "GTC");
        }

        // Only add reduceOnly if it's true
        if (reduceOnly) {
            payload["reduceOnly"] = true;
        }

        if (price > 0 && type == "LIMIT") {
            payload["price"] = price;
        }
        
        if (stopPrice > 0 && (type.find("STOP") != std::string::npos || type.find("TAKE_PROFIT") != std::string::npos)) {
            payload["stopPrice"] = stopPrice;
        }

        // Add unique client order ID for tracking
        std::string client_order_id = "beast_" + std::to_string(std::time(nullptr)) + "_" + 
                                    std::to_string(std::rand() % 10000);
        payload["newClientOrderId"] = client_order_id;
        
        // Log the ENTRY order before sending
        log_order(payload, "SENDING_ENTRY");
        
        // STEP 1: Send the main entry order
        std::string response_string;
        bool success = send_order_to_binance(payload.dump(), api_key, api_secret, response_string);
        
        if (!success) {
            log_error("Failed to send main entry order to Binance");
            return false;
        }

        try {
            json response_json = json::parse(response_string);
            
            // Check for Binance API errors in response
            if (response_json.contains("code") && response_json["code"] != 0) {
                std::string error_msg = response_json.value("msg", "Unknown Binance API error");
                log_error("Binance API error: " + error_msg, response_json["code"].get<int>(), payload.dump());
                log_order(response_json, "ERROR");
                return false;
            }
            
            // Log successful ENTRY order
            log_order(response_json, "ENTRY_EXECUTED");
            
            // Add to active trades
            add_active_trade(response_json);
            
            std::cout << "[BEAST] ✅ Entry order executed successfully!" << std::endl;
            
        } catch (const json::exception& e) {
            log_error("Error parsing Binance response: " + std::string(e.what()) + " - Response: " + response_string);
            return false;
        }

        // STEP 2: Place Stop Loss Order (if specified)
        if (stopLoss > 0.0) {
            std::cout << "[BEAST] 🛡️ Placing Stop Loss order at: $" << stopLoss << std::endl;
            bool sl_success = place_stop_loss_order(symbol, side, positionSide, quantity, stopLoss, api_key, api_secret);
            if (!sl_success) {
                log_warning("Failed to place Stop Loss order, but main entry order succeeded");
            }
        } else {
            std::cout << "[BEAST] ℹ️ No Stop Loss specified in signal" << std::endl;
        }

        // STEP 3: Place Take Profit Order (if specified)  
        if (takeProfit > 0.0) {
            std::cout << "[BEAST] 🎯 Placing Take Profit order at: $" << takeProfit << std::endl;
            bool tp_success = place_take_profit_order(symbol, side, positionSide, quantity, takeProfit, api_key, api_secret);
            if (!tp_success) {
                log_warning("Failed to place Take Profit order, but main entry order succeeded");
            }
        } else {
            std::cout << "[BEAST] ℹ️ No Take Profit specified in signal" << std::endl;
        }
        
        std::cout << "[BEAST] 🚀 Trade execution completed!" << std::endl;
        return true;
        
    } catch (const json::exception& e) {
        log_error("JSON parsing error in load_and_send_trade: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        log_error("Exception in load_and_send_trade: " + std::string(e.what()));
        return false;
    }
}

// Fixed version of add_active_trade function with proper type handling
void add_active_trade(const json& order) {
    try {
        std::vector<json> active_trades;
        std::ifstream file("active_trades.json");
        if (file.good()) {
            json temp_json;
            file >> temp_json;
            if (temp_json.is_array()) {
                active_trades = temp_json.get<std::vector<json>>();
            }
        }
        file.close();

        // Helper function to safely convert string to double
        auto safe_to_double = [](const json& value, double default_val = 0.0) -> double {
            if (value.is_string()) {
                try {
                    return std::stod(value.get<std::string>());
                } catch (...) {
                    return default_val;
                }
            } else if (value.is_number()) {
                return value.get<double>();
            }
            return default_val;
        };

        // Helper function to safely convert to timestamp
        auto safe_to_timestamp = [](const json& value) -> std::time_t {
            if (value.is_string()) {
                try {
                    return static_cast<std::time_t>(std::stoll(value.get<std::string>()));
                } catch (...) {
                    return std::time(nullptr);
                }
            } else if (value.is_number()) {
                return static_cast<std::time_t>(value.get<long long>());
            }
            return std::time(nullptr);
        };

        // Create new trade entry with safe type conversion
        json active_trade = {
            {"trade_id", order.value("clientOrderId", "unknown")},
            {"symbol", order.value("symbol", "unknown")},
            {"side", order.value("side", "unknown")},
            {"quantity", safe_to_double(order.value("origQty", "0"))},
            {"entry", safe_to_double(order.value("avgPrice", "0"))},
            {"timestamp", safe_to_timestamp(order.value("updateTime", std::time(nullptr)))},
            {"status", order.value("status", "NEW")},
            {"order_id", order.value("orderId", 0)}
        };

        // Add optional fields safely
        if (order.contains("stopLoss")) {
            active_trade["stopLoss"] = safe_to_double(order["stopLoss"]);
        }
        if (order.contains("takeProfit")) {
            active_trade["takeProfit"] = safe_to_double(order["takeProfit"]);
        }

        // Add cumulative quote (total cost) if available
        if (order.contains("cumQuote")) {
            active_trade["total_cost"] = safe_to_double(order["cumQuote"]);
        }

        // Add to vector
        active_trades.push_back(active_trade);

        // Write back to file
        std::ofstream outfile("active_trades.json");
        json json_array = active_trades;
        outfile << json_array.dump(2);
        outfile.close();

        // Add to OrderManager with safe conversion
        OrderManager::get_instance().add_trade(active_trade);
        
        std::cout << "[TRADE] ✅ Successfully logged trade: " 
                  << active_trade["symbol"] << " - " 
                  << active_trade["quantity"] << " @ $" 
                  << active_trade["entry"] << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Error adding active trade: " << e.what() << std::endl;
        // Log the original order data for debugging
        std::cerr << "[DEBUG] Original order JSON: " << order.dump(2) << std::endl;
    }
}

// Enhanced version of read_and_process_signal function
void read_and_process_signal(const std::filesystem::path& signal_path) {
    try {
        std::cout << "[BEAST] Processing signal: " << signal_path.string() << std::endl;
        
        // First, check if file exists and is readable
        if (!std::filesystem::exists(signal_path)) {
            log_error("Signal file does not exist: " + signal_path.string());
            return;
        }
        
        // Implement retry logic for more robust processing
        int max_retries = 3;
        int retry_count = 0;
        bool success = false;
        
        while (!success && retry_count < max_retries) {
            try {
                // Process the signal
                success = load_and_send_trade(signal_path.string());
                
                if (success) {
                    std::cout << "[BEAST] Successfully processed signal: " << signal_path.string() << std::endl;
                    
                    // Archive the processed signal
                    std::filesystem::path archive_dir = "./processed_signals";
                    std::filesystem::create_directories(archive_dir);
                    
                    // Create timestamped filename for archive
                    std::time_t now = std::time(nullptr);
                    char timestamp[32];
                    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));
                    
                    std::string archived_filename = signal_path.stem().string() + "_" + 
                                                 timestamp + signal_path.extension().string();
                    std::filesystem::path archive_path = archive_dir / archived_filename;
                    
                    // Copy to archive folder
                    std::filesystem::copy_file(signal_path, archive_path, 
                                              std::filesystem::copy_options::overwrite_existing);
                    
                    // Remove the original signal
                    std::filesystem::remove(signal_path);
                } else {
                    retry_count++;
                    if (retry_count < max_retries) {
                        std::cerr << "[BEAST] Failed to process signal: " << signal_path.string() 
                                  << ", retrying (" << retry_count << "/" << max_retries << ")" << std::endl;
                        std::this_thread::sleep_for(std::chrono::seconds(2 * retry_count));  // Exponential backoff
                    } else {
                        log_error("Failed to process signal after max retries: " + signal_path.string());
                        
                        // Move to error folder for inspection
                        std::filesystem::path error_dir = "./error_signals";
                        std::filesystem::create_directories(error_dir);
                        
                        // Create timestamped filename for error archive
                        std::time_t now = std::time(nullptr);
                        char timestamp[32];
                        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));
                        
                        std::string error_filename = signal_path.stem().string() + "_error_" + 
                                                   timestamp + signal_path.extension().string();
                        std::filesystem::path error_path = error_dir / error_filename;
                        
                        std::filesystem::copy_file(signal_path, error_path, 
                                                 std::filesystem::copy_options::overwrite_existing);
                        
                        // Remove original after copying to error folder
                        std::filesystem::remove(signal_path);
                    }
                }
            } catch (const std::exception& e) {
                log_error(std::string("Exception in retry loop: ") + e.what());
                retry_count++;
                std::this_thread::sleep_for(std::chrono::seconds(2 * retry_count));
            }
        }
    } catch (const std::exception& e) {
        log_error(std::string("Exception in signal processing: ") + e.what() + " - " + signal_path.string());
    }
}

int main() {
    std::cout << "============================\n";
    std::cout << "🔁  BEAST Engine Initializing\n";
    std::cout << "============================\n" << std::endl;

    // Create necessary directories
    std::filesystem::create_directories("./trade_signals");
    std::filesystem::create_directories("./logs");
    std::filesystem::create_directories("./processed_signals");
    std::filesystem::create_directories("./error_signals");

    // Initialize CURL globally
    curl_global_init(CURL_GLOBAL_ALL);

    // Add code to test the API connection
    std::cout << "🔍 Testing Binance API connection..." << std::endl;
    
    std::string api_key, api_secret;
    if (!get_api_credentials(api_key, api_secret)) {
        std::cerr << "❌ API credentials missing. Please check .env file or environment variables." << std::endl;
        std::cerr << "Create a .env file with:" << std::endl;
        std::cerr << "BINANCE_API_KEY=your_api_key_here" << std::endl;
        std::cerr << "BINANCE_API_SECRET=your_api_secret_here" << std::endl;
        return 1;
    } else {
        // Initialize dynamic precision system
        std::cout << "🔧 Loading symbol precision rules from Binance..." << std::endl;
        if (initialize_symbol_precision_from_binance(api_key)) {
            std::cout << "✅ Dynamic precision system initialized successfully!" << std::endl;
        } else {
            std::cout << "⚠️ Failed to load dynamic precision, using fallback hardcoded rules" << std::endl;
        }
        
        // Create a test request to check account info
        std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        std::string query_string = "timestamp=" + timestamp;
        
        // Use the trade executor's hmac generation
        std::string signature = hmac_sha256(api_secret, query_string);
        std::string url = "https://fapi.binance.com/fapi/v2/account?" + query_string + "&signature=" + signature;
        
        CURL* curl = curl_easy_init();
        if(curl) {
            std::string response;
            struct curl_slist *headers = NULL;
            headers = curl_slist_append(headers, ("X-MBX-APIKEY: " + api_key).c_str());
            
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            
            CURLcode res = curl_easy_perform(curl);
            if(res == CURLE_OK) {
                try {
                    json response_json = json::parse(response);
                    if (response_json.contains("code") && response_json["code"] != 0) {
                        std::cerr << "❌ API test failed: " << response_json["msg"] << std::endl;
                        log_error("API connection test failed: " + response_json["msg"].get<std::string>());
                    } else {
                        std::cout << "✅ API connection test successful!" << std::endl;
                        if (response_json.contains("totalWalletBalance")) {
                            std::cout << "💰 Account balance: " << response_json["totalWalletBalance"] << " USDT" << std::endl;
                        }
                    }
                } catch (const json::exception& e) {
                    std::cerr << "❌ Failed to parse API response: " << e.what() << std::endl;
                    log_error("Failed to parse API response: " + std::string(e.what()));
                }
            } else {
                std::cerr << "❌ API connection test failed: " << curl_easy_strerror(res) << std::endl;
                log_error("API connection test failed: " + std::string(curl_easy_strerror(res)));
            }
            
            curl_easy_cleanup(curl);
            curl_slist_free_all(headers);
        }
    }

    std::thread signal_watcher([]() {
        std::cout << "[BEAST] Watching ./trade_signals for signals...\n";
        while (true) {
            try {
                for (const auto& file : std::filesystem::directory_iterator("./trade_signals")) {
                    if (file.path().extension() == ".json") {
                        read_and_process_signal(file.path());
                    }
                }
            } catch (const std::exception& e) {
                log_error(std::string("Exception in signal watcher: ") + e.what());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        }
    });

    std::thread monitor_thread([]() {
        std::cout << "[BEAST] Starting trade monitor thread...\n";
        while (true) {
            try {
                monitor_trades();
            } catch (const std::exception& e) {
                log_error(std::string("Exception in monitor thread: ") + e.what());
            }
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    });

    // Wait for threads to complete (they won't under normal conditions)
    signal_watcher.join();
    monitor_thread.join();
    
    // Clean up CURL (this will never be reached in normal operation)
    curl_global_cleanup();
    
    return 0;
}