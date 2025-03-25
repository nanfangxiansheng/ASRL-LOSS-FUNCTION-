# To begin with

In the codes included ,the test.py includes ASRL 'model togther with GBRT test model.ASRL actually act as an Loss function of XGBoost regression model.In this Code repository ,we only provide code corresponded with the dataset of california housing price.More datasets (especially those from the UCI database website) can not be downloaded in this code repository because we do not have permission to upload these datasets. If you want to test more datasets using our code, you can visit the [UCI][https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant] website and download and import them yourself, or use the data package integrated in Python on the UCI website to import and perform preliminary data processing.

When using the data package integrated in Python.You need to download and install the UCI Python library ucimlrepo, then use the import command to observe the characteristics of the data and process it to make it compatible with the interface of the ASRL model.

```powershell
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
combined_cycle_power_plant = fetch_ucirepo(id=294) 
  
# data (as pandas dataframes) 
X = combined_cycle_power_plant.data.features 
y = combined_cycle_power_plant.data.targets 
  
# metadata 
print(combined_cycle_power_plant.metadata) 
  
# variable information 
print(combined_cycle_power_plant.variables) 

```

# Precautions

It is worth noting that in actual regression prediction, the predicted value each time is different, and the performance of the ASRL loss function is closely related to the values of the upper and lower quantile bounds. For different data samples, they should be set according to the distribution of these samples. In general, the upper quantile bound is between 0.1-0.3, and the lower quantile bound is between 0.7-0.9.
