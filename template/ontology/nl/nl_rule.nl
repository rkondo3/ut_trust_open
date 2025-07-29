- fundNumber: {{ comments['ut-trust:fundNumber'] }}
- brandName: {{ comments['ut-trust:brandName'] }}
- tradeDate: {{ comments['ut-trust:tradeDate'] }}
- valuationDate: {{ comments['ut-trust:valuationDate'] }}
- settlementDate: {{ comments['ut-trust:settlementDate'] }}
- unitPrice: {{ comments['ut-trust:unitPrice'] }}
- quantity: {{ comments['ut-trust:quantity'] }}
- grossAmount: {{ comments['ut-trust:grossAmount'] }}
- brokerageFee: {{ comments['ut-trust:brokerageFee'] }}
- brandCurrency: {{ comments['ut-trust:brandCurrency'] }}
- settlementAmountInBrandCurrency: {{ comments['ut-trust:settlementAmountInBrandCurrency'] }}
- settlementCurrency: {{ comments['ut-trust:settlementCurrency'] }}
- settlementAmountInSettlementCurrency: {{ comments['ut-trust:settlementAmountInSettlementCurrency'] }}

[Note]
There's following relationships between the items above:
1. quantity × unitPrice = grossAmount
2. grossAmount + brokerageFee = settlementAmount
, allowing for small rounding differences due to significant digits.
