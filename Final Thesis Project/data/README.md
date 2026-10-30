# Data

This study uses the **TravelPlanner** benchmark (Xie et al., 2024, arXiv:2402.01622),
released under CC-BY-4.0. It is not redistributed here (~358 MB). Obtain it from the
official project (https://github.com/OSU-NLP-Group/TravelPlanner) and arrange it as:

```
data/
  Query database/      train.csv  validation.csv  test.csv  (+ *_ref_info.jsonl)
  Tools database/
    accommodations/clean_accommodations_2022.csv
    attractions/attractions.csv
    flights/clean_Flights_2022.csv
    googleDistanceMatrix/distance.csv
    restaurants/clean_restaurant_2022.csv
    background/ (citySet.txt, citySet_with_states.txt, stateSet.txt)
```

Only `validation.csv` and the five tool tables are read by the code. To keep the data elsewhere,
set `THESIS_DATASET_ROOT=/path/to/folder/containing/the/two/directories`.

Sanity check: `python -c "from src import data_loader; print(len(data_loader.load_queries('validation')))"`
must print `180` (60 easy / 60 medium / 60 hard).
