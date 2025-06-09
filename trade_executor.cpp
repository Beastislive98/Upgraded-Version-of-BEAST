// trade_executor.cpp - Production Ready Version

#include <iostream>
#include <cstdlib>
#include <string>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <thread>
#include <openssl/hmac.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "trade_executor.hpp"
#include "order_manager.hpp"
#include "error_handler.hpp"
#include "order_manager.h"

using json = nlohmann::json;

// Callback function for CURL to capture response
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string hmac_sha256(const std::string &key, const std::string &data) {
    unsigned char* digest = NULL;
    unsigned int digestLen = 32;
    
    digest = (unsigned char*)HMAC(EVP_sha256(), 
                                 key.c_str(), key.length(),
                                 (unsigned char*)data.c_str(), data.length(),
                                 NULL, &digestLen);
    
    std::stringstream ss;
    for (unsigned int i = 0; i < digestLen; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }
    
    return ss.str();
}

bool send_order_to_binance(const std::string& payload, const std::string& api_key, 
                          const std::string& api_secret, std::string& response) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    try {
        // Parse the original payload
        json order_json = json::parse(payload);
        
        // Add timestamp (milliseconds since epoch)
        order_json["timestamp"] = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        // Convert to query string for signing
        std::string query_string;
        for (auto& el : order_json.items()) {
            if (!query_string.empty()) query_string += "&";
            
            std::string value_str;
            if (el.value().is_string())
                value_str = el.value().get<std::string>();
            else if (el.value().is_boolean())
                value_str = el.value().get<bool>() ? "true" : "false";
            else
                value_str = el.value().dump();
                
            // Remove quotes from numbers and other primitives
            value_str.erase(std::remove(value_str.begin(), value_str.end(), '\"'), value_str.end());
            
            query_string += el.key() + "=" + value_str;
        }
        
        // Generate signature
        std::string signature = hmac_sha256(api_secret, query_string);
        
        // Add signature to query string
        std::string final_query = query_string + "&signature=" + signature;
        
        // Use POST with parameters in REQUEST BODY (not URL)
        std::string url = "https://fapi.binance.com/fapi/v1/order";
        
        std::cout << "[DEBUG] Sending order to Binance: " << final_query << std::endl;
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("X-MBX-APIKEY: " + api_key).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, final_query.c_str());
        
        // Set timeout and retry options
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
        
        // Set function to capture response
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl);
        bool success = (res == CURLE_OK);

        // Log the EXACT response for debugging
        std::cout << "[DEBUG] Binance response: " << response << std::endl;

        // Process response if HTTP request was successful
        if (success) {
            try {
                json response_json = json::parse(response);
                process_order_response(response, order_json);
            } catch (const json::exception& e) {
                log_error("Failed to parse Binance response: " + std::string(e.what()) + ", Response: " + response);
                success = false;
            }
        } else {
            log_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
        }

        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
        return success;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in send_order_to_binance: " << e.what() << std::endl;
        log_error("Exception in send_order_to_binance: " + std::string(e.what()));
        curl_easy_cleanup(curl);
        return false;
    }
}

std::string build_order_payload(const json& trade_signal) {
    json payload;
    payload["symbol"] = trade_signal["symbol"];
    payload["side"] = trade_signal["side"];
    payload["type"] = trade_signal.value("type", "MARKET");
    
    // Handle quantity precision and minimum notional value
    double raw_quantity = trade_signal["quantity"].get<double>();
    std::string symbol = trade_signal["symbol"];
    
    // Basic precision handling
    int precision = 0;  // Default to whole numbers for most alt coins
    if (symbol.find("BTC") != std::string::npos || symbol.find("ETH") != std::string::npos) {
        precision = 3;
    } else if (symbol.find("BNB") != std::string::npos) {
        precision = 2;
    }
    
    double quantity = (precision > 0) ? 
        std::floor(raw_quantity * std::pow(10.0, precision)) / std::pow(10.0, precision) :
        std::floor(raw_quantity);
    
    if (quantity < 1.0 && precision == 0) quantity = 1.0;  // Minimum quantity
    
    // Note: For complete notional value handling, you would need to fetch current price here
    // For now, this is a basic implementation. The main logic is in BEAST.cpp
    
    payload["quantity"] = quantity;
    payload["positionSide"] = trade_signal.value("positionSide", "LONG");
    
    // Only add timeInForce for order types that require it (NOT for MARKET orders)
    std::string order_type = trade_signal.value("type", "MARKET");
    if (order_type != "MARKET") {
        payload["timeInForce"] = trade_signal.value("timeInForce", "GTC");
    }

    // Only add reduceOnly if it's true
    bool reduceOnly = trade_signal.value("reduceOnly", false);
    if (reduceOnly) {
        payload["reduceOnly"] = true;
    }

    if (trade_signal.contains("price") && trade_signal["price"].get<double>() > 0) {
        payload["price"] = trade_signal["price"];
    }
    
    if (trade_signal.contains("stopPrice") && trade_signal["stopPrice"].get<double>() > 0) {
        payload["stopPrice"] = trade_signal["stopPrice"];
    }
    
    // Add client order ID for tracking
    std::string client_order_id = "beast_" + std::to_string(std::time(nullptr)) + "_" + 
                                std::to_string(std::rand() % 10000);
    payload["newClientOrderId"] = client_order_id;

    return payload.dump();
}

// Add function to parse and use the execution parameters
bool send_order_with_dynamic_execution(const std::string& payload, const json& execution_params, 
                                      const std::string& api_key, const std::string& api_secret,
                                      std::string& response) {
    // Extract execution strategy
    std::string strategy = execution_params.value("strategy", "immediate");
    json params = execution_params.value("params", json::object());
    
    if (strategy == "immediate") {
        // Standard order execution - use existing code
        return send_order_to_binance(payload, api_key, api_secret, response);
    } 
    else if (strategy == "twap") {
        // TWAP execution
        int slices = params.value("slices", 1);
        double slice_size = params.value("slice_size", 0.0);
        int interval_seconds = params.value("interval_seconds", 60);
        
        json order_json = json::parse(payload);
        double total_quantity = order_json["quantity"].get<double>();
        double remaining = total_quantity;
        
        if (slice_size <= 0) {
            slice_size = total_quantity / slices;
        }
        
        bool overall_success = true;
        std::string combined_response = "{\"slices\":[";
        
        // Execute slices
        for (int i = 0; i < slices && remaining > 0; i++) {
            // Adjust quantity for this slice
            double slice_quantity = std::min(slice_size, remaining);
            order_json["quantity"] = slice_quantity;
            
            // Create unique client order ID for each slice
            order_json["newClientOrderId"] = "beast_twap_" + std::to_string(std::time(nullptr)) + 
                                          "_" + std::to_string(i) + "_" + std::to_string(std::rand() % 1000);
            
            // Send this slice
            std::string slice_response;
            bool success = send_order_to_binance(order_json.dump(), api_key, api_secret, slice_response);
            
            if (success) {
                combined_response += slice_response;
                if (i < slices - 1 && remaining - slice_quantity > 0) {
                    combined_response += ",";
                }
            } else {
                log_error("TWAP slice " + std::to_string(i+1) + " execution failed");
                overall_success = false;
                break;
            }
            
            // Update remaining quantity
            remaining -= slice_quantity;
            
            // Wait for the interval if not the last slice
            if (i < slices - 1 && remaining > 0) {
                std::this_thread::sleep_for(std::chrono::seconds(interval_seconds));
            }
        }
        
        combined_response += "]}";
        response = combined_response;
        return overall_success;
    }
    else if (strategy == "iceberg") {
        // Iceberg execution
        double visible_size = params.value("visible_size", 0.0);
        
        json order_json = json::parse(payload);
        double total_quantity = order_json["quantity"].get<double>();
        double remaining = total_quantity;
        
        if (visible_size <= 0) {
            visible_size = total_quantity * 0.1; // Default to 10% of total
            if (visible_size <= 0) visible_size = total_quantity;
        }
        
        bool overall_success = true;
        std::string combined_response = "{\"chunks\":[";
        int chunk_count = 0;
        
        // Execute iceberg chunks
        while (remaining > 0) {
            // Set visible quantity
            double chunk_size = std::min(visible_size, remaining);
            order_json["quantity"] = chunk_size;
            
            // Create unique client order ID for each chunk
            order_json["newClientOrderId"] = "beast_iceberg_" + std::to_string(std::time(nullptr)) + 
                                          "_" + std::to_string(chunk_count) + "_" + std::to_string(std::rand() % 1000);
            
            // Send this chunk
            std::string chunk_response;
            bool success = send_order_to_binance(order_json.dump(), api_key, api_secret, chunk_response);
            
            if (success) {
                combined_response += chunk_response;
                if (remaining - chunk_size > 0) {
                    combined_response += ",";
                }
            } else {
                log_error("Iceberg chunk " + std::to_string(chunk_count+1) + " execution failed");
                overall_success = false;
                break;
            }
            
            // Update remaining quantity
            remaining -= chunk_size;
            chunk_count++;
            
            if (remaining > 0) {
                // Wait a short time between chunks
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        
        combined_response += "]}";
        response = combined_response;
        return overall_success;
    }
    else {
        // Default to simple execution
        return send_order_to_binance(payload, api_key, api_secret, response);
    }
}

void process_order_response(const std::string& response, const json& original_trade) {
    try {
        // Parse the response
        json response_json = json::parse(response);
        
        // Log successful order
        if (response_json.contains("orderId")) {
            std::cout << "Order successfully placed. Order ID: " << response_json["orderId"] 
                      << " for " << original_trade["symbol"] << std::endl;
            
            // Create a combined object with both response and original trade information
            json combined_trade = response_json;
            
            // Add important fields from original trade that might not be in response
            if (original_trade.contains("stopLoss") && !combined_trade.contains("stopLoss")) {
                combined_trade["stopLoss"] = original_trade["stopLoss"];
            }
            
            if (original_trade.contains("takeProfit") && !combined_trade.contains("takeProfit")) {
                combined_trade["takeProfit"] = original_trade["takeProfit"];
            }
            
            if (original_trade.contains("pattern_id") && !combined_trade.contains("pattern_id")) {
                combined_trade["pattern_id"] = original_trade["pattern_id"];
            }
            
            // Update order status in order manager using the correct overloaded method
            OrderManager::get_instance().update_order_status(
                response_json["orderId"].get<long>(),
                response_json["status"].get<std::string>(),
                combined_trade);
                
            // Log order with its status
            log_order(combined_trade, "EXECUTED");
        } 
        // Handle error response
        else if (response_json.contains("code") && response_json["code"] != 0) {
            std::cerr << "Error placing order: " << response_json["msg"] 
                      << " (Code: " << response_json["code"] << ")" << std::endl;
            
            // Log error with detailed information
            log_error("Binance API error: " + response_json["msg"].get<std::string>(), 
                     response_json["code"].get<int>(), 
                     original_trade.dump());
                     
            // Log order with ERROR status
            log_order(response_json, "ERROR");
        }
        // Unexpected response format
        else {
            std::cerr << "Unexpected response format: " << response << std::endl;
            log_error("Unexpected Binance API response format", 0, response);
        }
    } 
    catch (const json::parse_error& e) {
        std::cerr << "Failed to parse response: " << e.what() << std::endl;
        std::cerr << "Raw response: " << response << std::endl;
        log_error("JSON parse error in process_order_response", 0, response);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in process_order_response: " << e.what() << std::endl;
        log_error("Exception in process_order_response", 0, e.what());
    }
}