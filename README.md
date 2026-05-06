# NBA Contract Estimator

Created by Zayn Remtulla.

This app estimates NBA annual contract value using season-before-signing production, efficiency, minutes, defensive impact, awards, availability, popularity, and nearby real contract comparisons.

## Run locally

```bash
cd ~/Downloads/nba_contract_estimator_final
python3 -m pip install -r requirements.txt
python3 -m streamlit run app/streamlit_app.py
```

## Notes

- Market tier is display/range context only. It does not force the estimate.
- Raw regression estimates from the full stats pattern.
- Local market compares the input player to nearby real contracts.
- Training stats should come from the season before the contract was signed.
