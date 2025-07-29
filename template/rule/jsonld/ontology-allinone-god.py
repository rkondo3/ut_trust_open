import re

def date_conversion(in_str):
    if in_str is None or str(in_str) == "No Information":
        return "No Information"
    return re.sub(r'\D', '', str(in_str))


def number_conversion(in_num):
    if in_num is None or str(in_num) == "No Information":
        return "No Information"
    elif isinstance(in_num, int) or isinstance(in_num, float):
       return abs(in_num)
    
    try:
        return abs(float(in_num))
    except:
        return "No Information (type error)"

def safe_get_nested(data, key1, key2, default="No Information"):
    """Safely get nested value, handling None at any level"""
    outer = data.get(key1)
    if outer is None:
        return default
    return outer.get(key2, default)

def safe_get_nested_with_fallback(data, primary_key, fallback_key, nested_key, default="No Information"):
    """Safely get nested value with fallback to another key"""
    primary = data.get(primary_key)
    if primary is not None:
        return primary.get(nested_key, default)
    
    fallback = data.get(fallback_key)
    if fallback is not None:
        return fallback.get(nested_key, default)
    
    return default

def convert(llm_output_dict):
    row = {
        "fund_code": llm_output_dict.get("fundNumber", "No Information"),
        "brand_name": llm_output_dict.get("brandName", "No Information"),
        "trade_date": date_conversion(llm_output_dict.get("tradeDate", llm_output_dict.get("valuationDate", "No Information"))),
        "settlement_date": date_conversion(llm_output_dict.get("settlementDate", "No Information")),
        "base_currency": safe_get_nested_with_fallback(llm_output_dict, "settlementAmountInBrandCurrency", "settlementAmountInSettlementCurrency", "currency"),
        "settlement_currency": safe_get_nested_with_fallback(llm_output_dict, "settlementAmountInSettlementCurrency", "settlementAmountInBrandCurrency", "currency"),
        "order_quantity": number_conversion(llm_output_dict.get("quantity", "No Information")),
        "unit_price": number_conversion(safe_get_nested(llm_output_dict, "unitPrice", "value")),
        "gross_amount": number_conversion(safe_get_nested(llm_output_dict, "grossAmount", "value")),
        "fee": number_conversion(safe_get_nested(llm_output_dict, "brokerageFee", "value", default=0)),
        "settlement_amount_base_currency": number_conversion(safe_get_nested_with_fallback(llm_output_dict, "settlementAmountInBrandCurrency", "settlementAmountInSettlementCurrency", "value")),
        "settlement_amount_settlement_currency": number_conversion(safe_get_nested_with_fallback(llm_output_dict, "settlementAmountInSettlementCurrency", "settlementAmountInBrandCurrency", "value")),
    }
    return row
