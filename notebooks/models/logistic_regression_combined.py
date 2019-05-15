# coding: utf-8

# # Logistic Regression Model

# ## Data Import And Cleaning

# In[1]:
print('############### {} ###############'.format('Data Import And Cleaning'))

import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import scripts.logistic_regression_functions as lrf
import scripts.helper_functions as hf
from functools import partial

# In[2]:


train_files = ['../../data/cleaned/utah_2017_vineyard.csv',
               '../../data/cleaned/utah_2017_marina.csv']

test_files = ['../../data/cleaned/utah_2018_marina.csv',
              '../../data/cleaned/utah_2018_bird.csv']

drop_columns = ['Chlorophyll ug/L', 'Chlorophyll RFU', 'Wiper Pos V',
                'Cable Pwr V', 'Battery V', 'Chlorophyll (ug/L)', 'FDOM RFU']

x_columns = ['Temp C', 'Sp Cond (uS/cm)', 'pH (mV)', 'pH', 'Turbidity (NTU)',
             'ODOSat%', 'ODO (mg/L)']

target_column = 'BGA-Phycocyanin RFU'

RFU_THRESHOLD = 1.2
train_index = 0
test_index = 0
# the operation to be used for math operations during training.
math_operation = partial(np.square)

# In[3]:


# Load the data
train_dfs = lrf.import_df_data(train_files, drop_columns)
test_dfs = lrf.import_df_data(test_files, drop_columns)

train_dfs[train_index].head()

# ## Use Datetime Index

# In[8]:
print('############### {} ###############'.format('Use Datetime Index'))

print(test_dfs[test_index].dtypes)
print(train_dfs[train_index].dtypes)

# In[9]:


# We will need to engineer a datetime column for our 2017 data

# Create a single datetime column from the strings provided in our csv's
for i in range(0, len(train_dfs)):
    timestamp = train_dfs[i]['Date (mm.dd.yyyy)'] + ' ' + train_dfs[i]['Time 24hr']
    timestamp = pd.to_datetime(timestamp)
    train_dfs[i]['timestamp'] = timestamp
    train_dfs[i]['datetime'] = timestamp
    train_dfs[i] = train_dfs[i].set_index('timestamp')
    train_dfs[i] = train_dfs[i].drop(columns=['Date (mm.dd.yyyy)', 'Time 24hr'])
train_dfs[train_index].head()

# In[10]:


# Now to datetime index our 2018 data
for i in range(0, len(test_dfs)):
    timestamp = pd.to_datetime(test_dfs[i]['Time America/Boise UTC-06:00'])
    test_dfs[i]['timestamp'] = timestamp
    test_dfs[i]['datetime'] = timestamp
    test_dfs[i] = test_dfs[i].set_index('timestamp')
    test_dfs[i] = test_dfs[i].drop(columns=['Time America/Boise UTC-06:00'])
test_dfs[test_index].head()

# ## Fill missing values
# 

# In[11]:
print('############### {} ###############'.format('Fill missing values'))

# Fill all missing values with the mean
for df in test_dfs + train_dfs:
    for column in df.columns:
        number_null = np.sum(df[column].isnull())
        column_length = len(df[column])
        print('{} percent null: {:.1f}%'.format(
            column, (number_null / column_length) * 100))
        if number_null > 0:
            print("Filling {} with mean: {}\n".format(column, df[column].mean()))
            df[column] = df[column].fillna(df[column].mean())

# ## Add relative columns

# In[12]:
print('############### {} ###############'.format('Add relative columns'))

relative_names = [col + '_relative' for col in x_columns]
for i in range(0, len(train_dfs)):
    train_dfs[i][relative_names] = train_dfs[i][x_columns] - train_dfs[i][x_columns].mean()
    test_dfs[i][relative_names] = test_dfs[i][x_columns] - test_dfs[i][x_columns].mean()
# x_columns = list(set(x_columns + relative_names))
train_dfs[train_index].head()

# ## Add Weather Data

# In[13]:
print('############### {} ###############'.format('Add Weather Data'))

## Import And Clean Weather Data
weather = pd.read_csv('../../data/cleaned/daily_weather_metric_2017_2018.csv')

# Find out how much of the data is missing for each column.
drop_columns = []
missing_threshold = .05
for column in weather.columns:
    number_null = np.sum(weather[column].isnull())
    column_length = len(weather[column])
    should_drop_column = (number_null / column_length) > missing_threshold
    print('{} percent null: {:.1f}%'.format(column, (number_null / column_length) * 100))
    if should_drop_column:
        drop_columns.append(column)
print('dropping columns: {}'.format(drop_columns))
weather = weather.drop(drop_columns, axis=1)

# Drop the columns we will not be using.
weather = weather.drop(columns=['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'TOBS'])

weather['DATE'] = pd.to_datetime(weather['DATE'])
# Set a datetime index for convenience
weather['timestamp'] = weather['DATE']
weather = weather.set_index('timestamp')
print(weather.dtypes)

weather['DATE'] = weather['DATE'].apply(lambda x: x.date())

# In[14]:


# Add the weather data to our training/testing sets
count = 0
total = len(train_dfs + test_dfs)
for df in train_dfs + test_dfs:
    count += 1
    print("starting dataframe {}/{}".format(count, total))
    for column in weather.drop(columns='DATE').columns:
        print("adding series for column:", column)
        values = []
        old_datetime = None
        value = None
        for d in df['datetime']:
            d = d.date()
            if d != old_datetime:
                old_datetime = d
                value = weather[weather['DATE'] == d][column]
            values.append(value.values[0] if len(value.values) > 0 else np.nan)
        df[column] = values

# ### Adding the Wind to weather
# 

# In[15]:
print('############### {} ###############'.format('Adding the Wind to weather'))

weather_files = ['../../data/raw_data/weather/provo_airport_2017', '../../data/raw_data/weather/provo_airport_2018']
# the array to store the readings in
contents = []

# Extract the weather data in the ISD format
for file in weather_files:
    contents.append(hf.extract_weather_data(file))

# In[16]:


# the dfs containing the wind information for the readings.
wind_frame_2017 = pd.DataFrame(columns=["angle", "speed"])
wind_frame_2018 = pd.DataFrame(columns=["angle", "speed"])

wind_frames = [wind_frame_2017, wind_frame_2018]
for i in range(len(contents)):
    wind_dict = {}
    for reading in contents[i]:
        reading.GPO_timestamp = hf.round_time(reading.GPO_timestamp)

        # no key value made yet
        if reading.GPO_timestamp not in wind_dict.keys():
            # check to see if the readings are valid
            # Sets them to 'invalid' so that when we inpute them it can be handled
            if reading.WO_wind_angle == '999':
                reading.WO_wind_angle = np.nan
            else:
                reading.WO_wind_angle = int(reading.WO_wind_angle)
            if reading.WO_wind_speed == '9999':
                reading.WO_wind_speed = np.nan
            else:
                reading.WO_wind_speed = int(reading.WO_wind_speed)
            wind_dict[reading.GPO_timestamp] = [reading.WO_wind_angle, reading.WO_wind_speed]
    wind_frames[i] = pd.DataFrame.from_dict(wind_dict, orient='index')
    wind_frames[i] = wind_frames[i].rename(columns={0: 'Wind Angle', 1: 'Wind Speed'})

# In[17]:


# we need to join the wind data now to the existing dataframes, but the dont care about

for i in range(len(train_dfs)):
    # train dfs is the 2017 data set so we need to set the wind frame to that 
    # dataset
    train_dfs[i] = train_dfs[i].join(wind_frames[0])
for i in range(len(test_dfs)):
    # test dfs is the 2018 data set so we need to set the wind frame to that
    # dataset
    test_dfs[i] = test_dfs[i].join(wind_frames[1])

for i in range(len(train_dfs)):
    train_dfs[i]['Wind Angle'] = train_dfs[i]['Wind Angle'].interpolate(limit_direction='both')
    train_dfs[i]['Wind Speed'] = train_dfs[i]['Wind Speed'].interpolate(limit_direction='both')
for i in range(len(test_dfs)):
    # test dfs is the 2018 data set so we need to set the wind frame to that dataset
    test_dfs[i]['Wind Angle'] = test_dfs[i]['Wind Angle'].interpolate(limit_direction='both')
    test_dfs[i]['Wind Speed'] = test_dfs[i]['Wind Speed'].interpolate(limit_direction='both')

# ## Extract Windows
# 

# In[19]:
print('############### {} ###############'.format('Extract Windows'))

presegmented_plot = pd.DataFrame(
    {'BGA RFU': train_dfs[train_index][target_column],
     'Bloom Threshold': np.full(train_dfs[train_index].count()[0], RFU_THRESHOLD)})

# Segment each data frame
percentiles = [0.0, 0.5, 1.0]
for i in range(0, len(train_dfs)):
    print("Windowizing 2017 data set:", i)
    train_dfs[i] = hf.windowize(
        train_dfs[i], 'datetime', target_column, feature_percentiles=percentiles)
    print("Windowizing 2018 data set:", i)
    test_dfs[i] = hf.windowize(
        test_dfs[i], 'datetime', target_column, feature_percentiles=percentiles)
    print()

# Update x_columns TODO REMOVE
to_drop = [target_column, 'datetime']
to_drop += ['datetime_{}'.format(p) for p in percentiles]
x_columns = train_dfs[0].drop(columns=to_drop).columns.values.tolist()
print("new x_columns:", x_columns)

# ## Bucket/Bin Features
# 
# ### Bin all numerical data

# In[21]:
print('############### {} ###############'.format('Bucket/Bin Features'))

bins = 3
drop_columns = []
new_columns = []
quantile_binning = False

# drop columns that don't have enough data variation to meet the required boundaries for binning.
for col in x_columns:
    for df in train_dfs + test_dfs:
        try:
            hf.bin_df(df[[col]], bins, quantile_binning)
        except ValueError:
            print("Could not bin {} with {} # of bins".format(col, bins))
            drop_columns.append(col)


for i in range(0, len(train_dfs)):
    temp = train_dfs[i][x_columns].drop(columns=drop_columns)
    binned, b_cols = hf.bin_df(temp, bins, quantile_binning)
    new_columns += b_cols
    train_dfs[i] = pd.concat([train_dfs[i], binned], axis='columns')

# add binned categories for testing sets
for i in range(0, len(test_dfs)):
    temp = test_dfs[i][x_columns].drop(columns=drop_columns)
    binned, b_cols = hf.bin_df(temp, bins, quantile_binning)
    new_columns += b_cols
    test_dfs[i] = pd.concat([test_dfs[i], binned], axis='columns')
print(test_dfs[test_index].dtypes)

# Add the new columns to x columns
x_columns = list(set(x_columns + new_columns))
print(x_columns)

# Add squared features
print('############### {} ###############'.format('Square Features'))
for i in range(len(train_dfs)):
    squared = lrf.apply_math_operation(train_dfs[i][x_columns], math_operation, "_squared")
    print("old train shape:", train_dfs[i].shape)
    train_dfs[i][squared.columns] = squared
    print("new train shape:", train_dfs[i].shape)

    squared = lrf.apply_math_operation(test_dfs[i][x_columns], math_operation, "_squared")
    print("old test shape:", test_dfs[i].shape)
    test_dfs[i][squared.columns] = squared
    print("new test shape:", test_dfs[i].shape)

# Add gradient features
print('############### {} ###############'.format('Gradient Features'))
for df in train_dfs + test_dfs:
    gradients = lrf.apply_math_operation(df[x_columns], np.gradient, '_gradient')
    df[gradients.columns] = gradients

# Add the new columns to x_columns
x_columns = list(set(x_columns + list(gradients.columns)))
x_columns = list(set(x_columns + list(squared.columns)))
print(x_columns)

# ## Add Weather Categories

# In[22]:
print('############### {} ###############'.format('Add Weather Categories'))

# Add a rainy categories
for df in test_dfs + train_dfs:
    for p in percentiles:
        df['rained_{}'.format(p)] = df['PRCP_{}'.format(p)].apply(
            lambda x: 1 if x > 0 else 0).astype('category')

# add the weather columns to our x_columns
# Update x_columns
to_drop = [target_column, 'datetime']
to_drop += ['datetime_{}'.format(p) for p in percentiles]
x_columns = list(set(x_columns + train_dfs[0].drop(columns=to_drop).columns.values.tolist()))
print("new x_columns:", x_columns)

# # ## Date Variables
#
# # In[23]:
# print('############### {} ###############'.format('Date Variables'))
#
#
# # Add month and day variables to our dataframes
# for df in train_dfs + test_dfs:
#     df['month'] = df['datetime'].apply(lambda x: x.month)
#     df['day'] = df['datetime'].apply(
#         lambda x: (x - datetime.datetime(x.year, 1, 1)).days)
#
# x_columns = list(set(x_columns + ['day', 'month']))
# train_dfs[train_index].head()

# ## Add Time of day Category

# In[24]:
print('############### {} ###############'.format('Add Time of day Category'))

for df in train_dfs + test_dfs:
    df['time of day'] = df['datetime'].apply(hf.create_time_of_day).astype('category')
x_columns.append('time of day')
train_dfs[train_index].head()

# ## Logistic Regression Model
# ### Null Model

# In[26]:
print('############### {} ###############'.format('Logistic Regression Model'))
print('############### {} ###############'.format('Null Model'))

lrf.add_target_column(train_dfs + test_dfs, threshold=RFU_THRESHOLD)

# In[27]:


# The null model's performance
max_iter = 25000
loss = "log"
model = SGDClassifier(max_iter=max_iter, loss=loss)
accuracy, recall, precision, cm, _, _, _ = lrf.train_model(
    model, train_dfs[train_index], test_dfs[test_index], x_columns, 'bloom', null_model=True)
print("Accuracy", accuracy)
print("Recall:", recall)
print("Precision", precision)
print("Confusion Matrix:\n", cm)
print("columns:", train_dfs[train_index].columns)

# ### All variables model

# In[28]:
print('############### {} ###############'.format('All variables model'))

# All variables model performance
max_iter = 25000
loss = "log"
model = SGDClassifier(max_iter=max_iter, loss=loss)
accuracy, recall, precision, cm, _, _, _ = lrf.train_model(
    model, train_dfs[train_index], test_dfs[test_index], x_columns, 'bloom')
print("Accuracy", accuracy)
print("Recall:", recall)
print("Precision", precision)
print("Confusion Matrix:\n", cm)
print("columns:", x_columns)

# ### Greedy Model

# In[30]:
print('############### {} ###############'.format('Greedy Model'))

max_iter = 25000
loss = "log"
model = SGDClassifier(max_iter=max_iter, loss=loss)
# Sort columns by accuracy
sorted_columns = lrf.sort_columns_by_metric(model, train_dfs[train_index],
                                            test_dfs[test_index],
                                            x_columns,
                                            'bloom')

# In[31]:


# create greedy model
model = SGDClassifier(max_iter=max_iter, loss=loss)

accuracy, recall, precision, cm, predictions, predictions_prob, model = lrf.greedy_model(
    model, train_dfs[train_index], test_dfs[test_index], x_columns,
    'bloom', sorted_columns)

# In[32]:


# Print the ROC curve.
predictions = [x[1] for x in predictions_prob]
lrf.roc_plot(test_dfs[test_index][['bloom']].values, predictions)

# ## Random Forest Model

# In[34]:
print('############### {} ###############'.format('Random Forest Model'))

# All Inputs
model = RandomForestClassifier(n_estimators=100)
accuracy, recall, precision, cm, predictions, predictions_prob, model = lrf.train_model(
    model, train_dfs[train_index], test_dfs[test_index], x_columns, 'bloom')
print("Accuracy", accuracy)
print("Recall:", recall)
print("Precision", precision)
print("Confusion Matrix:\n", cm)

# In[35]:


# Print the ROC curve.
predictions = [x[1] for x in predictions_prob]
lrf.roc_plot(test_dfs[test_index][['bloom']].values, predictions)

# ## Greedy Random Forest Model

# In[38]:
print('############### {} ###############'.format('Greedy Random Forest Model'))

model = RandomForestClassifier(n_estimators=100)
sorted_columns = lrf.sort_columns_by_metric(model, train_dfs[train_index],
                                            test_dfs[test_index],
                                            x_columns,
                                            'bloom')
accuracy, recall, precision, cm, predictions, predictions_prob, model = lrf.greedy_model(
    model, train_dfs[train_index], test_dfs[test_index], x_columns,
    'bloom', sorted_columns)
