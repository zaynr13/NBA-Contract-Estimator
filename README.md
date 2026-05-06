# NBA Contract Estimator

Created by Zayn Remtulla.

This app estimates NBA annual contract value using season-before-signing production, efficiency, minutes, defensive impact, awards, availability, popularity, and nearby real contract comparisons.

I built an NBA Contract Estimator that uses season-before-signing stats, defensive impact, awards, availability, and comparable recent contracts to estimate fair annual contract value.

Live app: https://nba-contract-estimator.streamlit.app/

## Notes

- Market tier is display/range context only. It does not force the estimate.
- Raw regression estimates from the full stats pattern.
- Local market compares the input player to nearby real contracts.
- Training stats should come from the season before the contract was signed.
