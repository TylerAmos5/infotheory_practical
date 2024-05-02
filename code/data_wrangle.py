from ucimlrepo import fetch_ucirepo 
import pandas as pd 

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data 
df = heart_disease.data.features 
# target
num_vector = heart_disease.data.targets 

# Cast num_vector to a numpy array 
num_vector = num_vector.iloc[:, 0]  # This selects the first column as a Series

# Convert vector to a Pandas Series with matching index 
num_series = pd.Series(num_vector, index=df.index)

df['num'] = num_series

#print(df.head())
# Save the dataset
df.to_csv('./data/heart_disease.csv',)

