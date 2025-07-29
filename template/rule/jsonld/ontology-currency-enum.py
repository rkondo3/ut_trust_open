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
    

def extract_currency_from_enum(enum_str):
    if enum_str is None or str(enum_str) == "No Information":
        return "No Information"
    return re.sub(r'ut-trust:', '', str(enum_str))

def safe_get_nested(data, *keys, default="No Information"):
    """Safely traverse nested dictionaries with any number of keys"""
    current = data
    for key in keys:
        if current is None or not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default

def safe_get_nested_with_fallback(data, primary_path, fallback_path, default="No Information"):
    """
    Safely get nested value with fallback path
    primary_path and fallback_path should be tuples of keys
    """
    result = safe_get_nested(data, *primary_path, default=None)
    if result is not None and result != "No Information":
        return result
    return safe_get_nested(data, *fallback_path, default=default)

def convert(llm_output_dict):
    row = {
        "fund_code": llm_output_dict.get("fundNumber", "No Information"),
        "brand_name": llm_output_dict.get("brandName", "No Information"),
        "trade_date": date_conversion(llm_output_dict.get("tradeDate", llm_output_dict.get("valuationDate", "No Information"))),
        "settlement_date": date_conversion(llm_output_dict.get("settlementDate", "No Information")),
        "base_currency": extract_currency_from_enum(
            safe_get_nested_with_fallback(
                llm_output_dict,
                ("settlementAmountInBrandCurrency", "ut-trust:currency", "@id"),
                ("settlementAmountInSettlementCurrency", "ut-trust:currency", "@id")
            )
        ),
        "settlement_currency": extract_currency_from_enum(
            safe_get_nested_with_fallback(
                llm_output_dict,
                ("settlementAmountInSettlementCurrency", "ut-trust:currency", "@id"),
                ("settlementAmountInBrandCurrency", "ut-trust:currency", "@id")
            )
        ),
        "order_quantity": number_conversion(llm_output_dict.get("quantity", "No Information")),
        "unit_price": number_conversion(safe_get_nested(llm_output_dict, "unitPrice", "value")),
        "gross_amount": number_conversion(safe_get_nested(llm_output_dict, "grossAmount", "value")),
        "fee": number_conversion(safe_get_nested(llm_output_dict, "brokerageFee", "value", default=0)),
        "settlement_amount_base_currency": number_conversion(
            safe_get_nested_with_fallback(
                llm_output_dict,
                ("settlementAmountInBrandCurrency", "value"),
                ("settlementAmountInSettlementCurrency", "value")
            )
        ),
        "settlement_amount_settlement_currency": number_conversion(
            safe_get_nested_with_fallback(
                llm_output_dict,
                ("settlementAmountInSettlementCurrency", "value"),
                ("settlementAmountInBrandCurrency", "value")
            )
        ),
    }
    return row
