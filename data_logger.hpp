// data_logger.hpp

#ifndef DATA_LOGGER_HPP
#define DATA_LOGGER_HPP

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Archive trade data to a file with timestamp
bool archive_trade_data(const json& trade, const std::string& filename_prefix = "executed_trade");

#endif // DATA_LOGGER_HPP