# Data Sources

Do not commit large data files to this repo.

## Download Links

| File | Source | Link |
|------|--------|------|
| WHO_COVID19_cleaned.csv | WHO COVID-19 Dashboard | https://data.who.int/dashboards/covid19/data |
| WHO_COVID19_cleaned_biweekly.csv | Derived from above | Run `work/data_biweekly_conversion.ipynb` |
| AUS_merged_nndss_all_diseases.xlsx | Australian NNDSS | https://www.health.gov.au/resources/collections/nndss-fortnightly-reports |
| AUS_rsv.xlsx | Extracted from NNDSS | Run `work/data_extract_aus_biweekly.ipynb` |
| AUS_measles.xlsx | Extracted from NNDSS | Run `work/data_extract_aus_biweekly.ipynb` |
| population.csv | World Bank | https://github.com/datasets/population |
| Singapore data | Singapore MOH | https://data.gov.sg/datasets?query=infectious+disease+bulletin |

## Reproduction
Run the notebooks in `work/` in this order:
1. `data_cleaning_who_covid.ipynb` — produces WHO_COVID19_cleaned.csv
2. `data_biweekly_conversion.ipynb` — produces WHO_COVID19_cleaned_biweekly.csv
3. `data_cleaning_aus_nndss.ipynb` — processes NNDSS Excel files
4. `data_extract_aus_biweekly.ipynb` — extracts RSV, Measles subsets
