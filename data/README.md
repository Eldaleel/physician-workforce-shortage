# Data notes

`final_dataset_model_ready.csv` is the processed country-year modelling table used by the model-building notebook. It contains 612 observations covering 159 countries from 2018 to 2022.

The raw World Bank downloads are not included in this portfolio package. To run the exploration and preparation notebooks, place CSV exports with four metadata rows and the following filenames in this directory:

| File | World Bank indicator |
|---|---|
| `physicians.csv` | Physicians (per 1,000 people) |
| `GDP.csv` | GDP (current US$) |
| `health_expenditure.csv` | Current health expenditure (% of GDP) |
| `population_growth.csv` | Population growth (annual %) |
| `school_enrollment.csv` | School enrollment, tertiary (% gross) |
| `unemployment.csv` | Unemployment, total (% of total labor force), national estimate |
| `urban_population.csv` | Urban population (% of total population) |

Source: [World Bank Open Data](https://data.worldbank.org/).

The `shortage` target equals 1 when a country is in the bottom 30% of physician density within its observation year and 0 otherwise.
