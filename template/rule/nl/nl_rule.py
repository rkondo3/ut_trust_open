import re

def date_conversion(in_str):
    if str(in_str) == "No Information":
        return in_str
    return re.sub(r'\D', '', str(in_str))


def number_conversion(in_num):
    if str(in_num) == "No Information":
        return in_num
    elif isinstance(in_num, int) or isinstance(in_num, float):
       return abs(in_num)
    try:
        return abs(float(in_num))
    except:
        return "No Information (type error)"

def convert(llm_output_dict):
    row = {
        "fund_code": llm_output_dict.get("fundNumber", "No Information"),
        "brand_name": llm_output_dict.get("brandName", "No Information"),
        "trade_date": date_conversion(llm_output_dict.get("tradeDate", llm_output_dict.get("valuationDate", "No Information"))),
        "settlement_date": date_conversion(llm_output_dict.get("settlementDate", "No Information")),
        "base_currency": llm_output_dict.get("brandCurrency", llm_output_dict.get("settlementCurrency", "No Information")),
        "settlement_currency": llm_output_dict.get("settlementCurrency", llm_output_dict.get("brandCurrency", "No Information")),
        "order_quantity": number_conversion(llm_output_dict.get("quantity", "No Information")),
        "unit_price": number_conversion(llm_output_dict.get("unitPrice", "No Information")),
        "gross_amount": number_conversion(llm_output_dict.get("grossAmount", "No Information")),
        "fee": number_conversion(llm_output_dict.get("brokerageFee", 0)),
        "settlement_amount_base_currency": number_conversion(llm_output_dict.get("settlementAmountInBrandCurrency", llm_output_dict.get("settlementAmountInSettlementCurrency", "No Information"))),
        "settlement_amount_settlement_currency": number_conversion(llm_output_dict.get("settlementAmountInSettlementCurrency", llm_output_dict.get("settlementAmountInBrandCurrency", "No Information"))),
    }
    return row