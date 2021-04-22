import pandas as pd
import numpy as np
from dictances import bhattacharyya as bha
from sklearn.neighbors import LocalOutlierFactor as LOF
import math
import os
import pickle
import json
import copy


def check_hours(df):
    """"Function to double check if hours are complete, delete hours with more than 3 zeroes"""

    minutes = ['00:00', '05:00', '10:00', '15:00', '20:00', '25:00', '30:00', '35:00', '40:00', '45:00', '50:00',
               '55:00']
    for date in df['timestamp'].dt.date.unique():
        current_day = df[df['timestamp'].dt.date == date]
        for hour in current_day['timestamp'].dt.hour.unique():
            current_hour = current_day[current_day['timestamp'].dt.hour == hour]
            if (len(current_hour) != 12):
                df.drop(current_hour.index, inplace=True)
            try:
                if current_hour['cars'].value_counts()[0.0] > 3:
                    df.drop(current_hour.index, inplace=True)
            except Exception as e:
                continue
    return df


def read_folder(current_intersection, configs, trac, direc):
    """Function to read all csv files (which is one per month) in the path and create one df from it."""

    print("Starting intersection: " + str(current_intersection))  # note which intersection its working on
    path = configs['data_folder'] + current_intersection + "//"

    df = pd.DataFrame(columns=['timestamp', 'cars'])  # create df to save everything in
    # loop through all files that end with csv:
    for file in os.listdir(path):  # read all files:
        if file.endswith(".csv"):  # for all csv files in the folder

            current_month = pd.read_csv(path + file, delimiter=";")
            cols = configs["trajectories"][trac][direc][current_intersection] + [
                current_intersection]  # define interesting columns
            current_month = current_month[cols]  # only keep interesting columns
            # some cleaning:
            current_month = current_month[:-1]  # last row is totals
            current_month = current_month.fillna(0)  # fill NA values with 0

            for sensor in configs['trajectories'][trac][direc][current_intersection]:
                current_month[sensor] = current_month[sensor].apply(
                    lambda x: x if x <= 600 else 0)  # remove sensor errors
                current_month[sensor] = current_month[sensor].loc[
                    current_month[sensor].shift(4) != current_month[sensor]]
            current_month[configs['trajectories'][trac][direc][current_intersection]] = current_month[
                configs['trajectories'][trac][direc][current_intersection]].clip(-1,
                                                                                 401)  # clip values between 0 and 400

            # now manipulate:
            current_month['cars'] = current_month[configs['trajectories'][trac][direc][current_intersection]].sum(
                axis=1)  # sum of all interesting columns
            current_month = current_month[[current_intersection, "cars"]]  # only keep name and total amount of cars
            current_month.columns = ['timestamp', 'cars']  # rename to timestamp for general format

            df = pd.concat([df, current_month])  # add to base df

    df['timestamp'] = pd.to_datetime(df['timestamp'])  # format as dt
    df['cars'] = df['cars'].clip(-1, len(configs['trajectories'][trac][direc][
                                             current_intersection] * 150))  # no intersection could be able to process sensors*150 cars
    df = df.sort_values(
        by='timestamp')  # ,ignore_index=True) #sort by timestamp #apparently ignore index does not work (anymore?)
    df = df.dropna()  # extra check to drop na values
    df = df.loc[
        (df['timestamp'] > '2014-12-31') & (df['timestamp'] < '2020-05-31')]  # delete faulty datapoints outside scope

    df = df.reset_index(drop=True)
    df = check_hours(df)  # try this afterwards
    return df


def fpd(df, configs, intersection, hours=1):
    """Function to aggregate a df of traffic info into one with fpds per window of 
    x hours (this means 12*hours values)"""
    freq = str(hours) + "H"
    aggregate = df.groupby(pd.Grouper(freq=freq, key='timestamp')).sum()  # aggregate by 1 hour
    df = pd.merge(df, aggregate, on='timestamp', how='left')  # merge with normal df
    df = df.fillna(method='ffill')  # fill with previous number
    df.columns = ['timestamp', 'cars', 'total']
    df['cars'][df['cars'] < 0] = 0  # some inconistencies in the data where cars could be negative
    df['total'][df[
                    'total'] < 0] = 1  # some inconsistencies in the data where total cars could be negative, set to 1 to avoid problems
    df['prob'] = df['cars'] / df['total']

    return df


def create_timeslot_array(data, name, window=12):
    """Function to reshape into numpy array shaped like (samples,window); e.g. 120 datapoints/12 (60min/5mins=12) = 10 FPDs.
    This is neccesary to create the bhattacharyya matrices. Misfunctions when an hour in the data has more or fewer than 12 datapoints (happens with double timestamps or missing data)
    Should be fixed by adding better data protection in the read_data function & rerunning the vlogbroker to output raw sensor values."""

    data['weekday'] = data['timestamp'].apply(lambda x: x.weekday())
    data['hour'] = data['timestamp'].apply(lambda x: x.hour)
    data_array = []  # init empty array that will contain 168 datasets with data for each hour on each weekday
    for i in range(7):
        timeslots = []
        for hour in range(24):
            try:
                datapoint = data[(data['weekday'] == i) & (data['hour'] == hour)]
                datapoint = data[(data['weekday'] == i) & (data['hour'] == hour)]
                x = np.array(datapoint['prob'])
                x = x.reshape(int(int(len(x)) / window),
                              window)  # data should be complete and divisible by 12, otherwise it fails.
                dates = sorted(
                    set(datapoint['timestamp'].apply(lambda x: x.floor(freq='H'))))  # add in hourly timestamp
                timeslots.append([x, dates])
            except Exception as e:
                print(e)
        data_array.append(timeslots)
    # output structure: data_array[7weekdays][24hours]; e.g. data_array[0][9] is data for monday mornings 9 am.
    return data_array


def create_matrices(inputs, intersection):
    """Function to create an array with n*n matrix with bha dist for each weekday from an array with 
    data for each weekday and the name of the intersection"""
    """input example: array[input[0:7][[0:24]-[data,dates]"""
    all_data = []
    print("Now starting to create bha matrices for intersection: " + intersection)
    for weekday in inputs:
        for timeslot in weekday:
            data = timeslot[0]  # timeslot[0]= data, timeslot[1]=timestamps
            m = dict()  # init dict with values
            values = dict(enumerate([dict(enumerate(x)) for x in data]))
            for hour in values:
                for hour2 in values:
                    try:
                        if hour in m:
                            m[hour].append(
                                bha(values[hour], values[hour2]) ** 2)  # values are squared for the SK learn algo
                        else:
                            m[hour] = [(bha(values[hour], values[hour2])) ** 2]
                    except:
                        m[hour] = 0

            # manually set difference to 0 for m[i][i]:
            for i in range(0, len(data)):
                try:
                    m[i][i] = 0
                except:
                    return m

            # now put this into a numpy array for the algo:
            x = np.array([m[v] for v in m])
            x = np.nan_to_num(x)
            all_data.append((x, timeslot[1]))  # append data with tuple of matrix,dates

    return all_data


def perform_lof(intersection_matrix, name):
    results = {}
    only_outliers = []
    outlier_count = 0
    total_length = 0
    for i in range(len(intersection_matrix)):  # for each weekday; 0 = monday 00:00, 168 = sunday 23:00
        datas = intersection_matrix[i][0]  # data is here
        dates = intersection_matrix[i][1]  # timeslots
        outliers, lof_scores = lof(datas)
        outlier_dates = [dates[x] for x in outliers]
        only_outliers.append([outliers, outlier_dates])
        outlier_count += len(outliers)
        total_length += len(lof_scores)
        if len(lof_scores) == len(dates):
            results[i] = pd.DataFrame({'index': dates, name: lof_scores})  # return dates and LOF scores
            results[i] = results[i].set_index(pd.DatetimeIndex(results[i]['index']))
            results[i] = results[i].drop(columns=['index'])
            results[i] = results[i].sort_values(by=name, ascending=False)
            print("Outliers in {}: {} out of {} data points.".format(name, str(outlier_count), str(total_length)))
        else:
            print('Problem with data in {}'.format(name))

    return only_outliers, results


def lof(data):
    if len(data) < 10:
        return [[0], [0]]
    model = LOF(n_neighbors=int(len(data) * 0.8), metric='precomputed', contamination='auto', n_jobs=-1)
    preds = model.fit_predict(data)  # perform LOF on all data
    outliers = np.where(preds == -1)[0]  # the indices of the outliers, use to compare with dates
    lof_scores = model.negative_outlier_factor_
    lof_scores = [x * -1 for x in lof_scores]
    return [outliers, lof_scores]


def create_lof_df(fpd_output, lof_output):
    """create df with all LOF scores"""
    df = pd.DataFrame()
    try:
        for intersection in fpd_output.keys():
            dates = copy.deepcopy(fpd_output[intersection])
            dates = dates.drop(columns=['weekday', 'hour'])
            dates = dates.groupby(
                pd.Grouper(key='timestamp', freq="H")).sum()  # extract hourly timeslots which have data
            dates[intersection] = np.nan
            dates = dates.drop(columns=['cars', 'total', 'prob'])
            for x in lof_output[intersection][1]:
                timeslot = copy.deepcopy(lof_output[intersection][1][x])
                dates.update(timeslot)
            df = pd.merge(df, dates, left_index=True, right_index=True, how='outer')
    except Exception as e:
        print(e)
        return df, lof_output
    return df


def create_historical_outlier_dfs():
    """Uses all functions above to return 'results'- output with outlier comparison dfs and the intermediate data which was processed.
    Input: list of intersection names (e.g.:'196003'). Will read data from folder in config file, list of intersections should also be there."""
    with open(r'configs.json', 'r') as f:
        configs = json.load(f)
    final_results = {}
    for trajectory in configs['trajectories']:
        final_results[trajectory] = {}
        for direction in configs['trajectories'][trajectory].keys():
            # first read the raw data from the pickle files:
            raw_data = {}
            for intersection in configs['trajectories'][trajectory][direction]:
                raw_data[intersection] = read_folder(intersection, configs, trajectory, direction)

            # then create FPDs:
            fpds = {}
            for intersection in raw_data:
                fpds[intersection] = fpd(raw_data[intersection], configs, intersection)
            # now create Bha matrices:
            matrices = {}
            for intersection in fpds:
                fpds_processed = create_timeslot_array(fpds[intersection], intersection)
                matrices[intersection] = create_matrices(fpds_processed, intersection)
            # Perform LOF:
            lof = {}
            for intersection in matrices:
                lof[intersection] = perform_lof(matrices[intersection], intersection)
            # Now build LOF df
            lof_df = create_lof_df(fpds, lof)

            approach_results = {'raw': raw_data, 'fpds': fpds, 'matrices': matrices, 'lof': lof, 'lof_df': lof_df}
            final_results[trajectory][direction] = approach_results
            print("Finished approach " + trajectory)
            pickle.dump(approach_results, open("resultsT{}_part_{}.pickle".format(trajectory, direction), "wb"))
    print("All finished.")
    return final_results
