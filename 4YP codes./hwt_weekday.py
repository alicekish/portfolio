# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:09:34 2022

@author: alice
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import datetime
import random
from scipy.optimize import curve_fit

# Hot Water Tank (HWT) script. This script provides model parameters to
# simulate heating and cooling of the water in the tank, as well as identifying
# times when hot water is required.

# This script returns:
    # a HEAT RATE in deg/sample (sample is 30 mins)
    # parameter for modelling passive cooling behaviour
    # Hot water usage times

##########################################################################

# Set usage for script
use_real_data = False
# False will mean the test data set is used instead. 

##########################################################################

# Get Data from HWT file
# from practice file

hwt_raw = pd. read_csv(r'...\Sample Data\Temp_log.csv', header=None)

##########################################################################

# Clean data to remove any anomalies
hwt_df = pd.DataFrame(hwt_raw)

hwt_df[0] = hwt_df[0].str[9:28] #time
hwt_df[1] = hwt_df[1].str[15:20] #collector
hwt_df[2] = hwt_df[2].str[9:14] #top
hwt_df[3] = hwt_df[3].str[12:17] #bottom
hwt_df[4] = hwt_df[4].str[8:13] #in
hwt_df[5] = hwt_df[5].str[9:11] #out


hwt_df.iloc[:,1:5] = hwt_df.iloc[:,1:5].astype(float)

# this trimming is specific to the test file. Live file probably would 
# work with this amount too as just takes most recent few weeks
hwt_sample = hwt_df[-(len(hwt_df)-268000):]

# only collect weekday data


hwt_sample = hwt_sample.reset_index(drop=True)
times = pd.to_datetime(hwt_sample[0])

to_remove = []
times= hwt_sample[0]

for each in times:
    if not str(each)[0:5] == '2022-':
        to_remove.append(each)
        
for each in to_remove:
    hwt_sample = hwt_sample.drop(hwt_sample.index[hwt_sample[0] == each])
    

##########################################################################

# Resample Top_temp and Bottom_temp Data to 30 minute intervals

# sample bottom temperature
index_btm = pd.to_datetime(hwt_sample[0])
values_btm = list(hwt_sample[3])
series_btm = pd.Series(values_btm, index=index_btm)

sampled_btm = series_btm.resample('30T').mean()

sampled_btm1 = sampled_btm[1:48*5+1]
sampled_btm2 = sampled_btm[48*7+1:48*12+1]
sampled_btm3 = sampled_btm[48*14+1:48*19+1]
sampled_btm4 = sampled_btm[48*21+1:]

#sample top temperature
index_top = pd.to_datetime(hwt_sample[0])
values_top = list(hwt_sample[2])
series_top = pd.Series(values_top, index=index_top)

sampled_top = series_top.resample('30T').mean()

sampled_top1 = sampled_top[1:48*5+1]
sampled_top2 = sampled_top[48*7+1:48*12+1]
sampled_top3 = sampled_top[48*14+1:48*19+1]
sampled_top4 = sampled_top[48*21+1:]


##########################################################################
# make some nice plots for report

day_top = sampled_top[48*3+1:48*5]
day_btm = sampled_btm[48*3+1:48*5]

times = [str(d)[5:-3]for d in day_top.index]

plt.figure(0)
plt.plot(day_top.index, day_top, label='Top temperature')
plt.plot(day_btm.index, day_btm, label='Bottom temperature')
plt.title('Temperatures in Hot Water Tank')
plt.xlabel('Time, mm-dd HH:MM')
plt.ylabel('Temperature, degrees Centigrade')
plt.xticks(ticks=day_top.index[0::4], labels=times[0::4] , rotation=90)
plt.legend(loc='lower right')

##########################################################################

# Identify HEATING and COOLING events

def find_points(start,end):    
    
    day_top = sampled_top[start:end]
    day_btm = sampled_btm[start:end]
    
    plt.figure(1)
    peaks_top, _ = find_peaks(day_top, height=0, distance=4, prominence=0.8)
    troughs_top, _ = find_peaks(-day_top, distance=4, prominence=0.8)
    plt.plot(day_top.index, day_top)
    plt.plot(day_top.index[peaks_top], day_top[peaks_top], 'x', color='green')
    plt.plot(day_top.index[troughs_top], day_top[troughs_top], 'x', color='red')
    plt.title('Peaks and troughs in Top of Tank Temperature')
    plt.xlabel('Time, mm-dd HH')
    plt.ylabel('Temperature, degrees Centigrade')
    plt.xticks(rotation=90)
    
    plt.figure(2)
    peaks_btm, _ = find_peaks(day_btm, height=0, distance=4, prominence=0.8)
    troughs_btm, _ = find_peaks(-day_btm, distance=4, prominence=0.8)
    plt.plot(day_btm.index, day_btm)
    plt.plot(day_btm.index[peaks_btm], day_btm[peaks_btm], 'x', color='green')
    plt.plot(day_btm.index[troughs_btm], day_btm[troughs_btm], 'x', color='red')
    plt.title('Peaks and troughs in Bottom of Tank')
    plt.xticks(rotation=90)
    
    return start, end, peaks_top, troughs_top, peaks_btm, troughs_btm

# to get HEATING events, want trough to peak in TOP temperature
# to get COOLING events, want peak to trough in TOP temperature

# this function returns a list of peaks and troughs, as well as a heat
# and cool df

def get_event_samples(start,end):
    
    day_top = sampled_top[start:end]
    
    peaks = list(find_points(start,end)[2])
    troughs = list(find_points(start,end)[3])
    
    points = peaks + troughs
    points.sort()
    
    # combine peaks/troughs list such that they must alternate. Remove
    # any 'excess' peaks/troughs that don't alternate.
    flag = False
    no_points = len(points)
    to_delete = []
    
    for each in list(range(1,no_points-1)):
        if points[each] in peaks and points[each-1] in peaks:
            flag = True
        elif points[each] in troughs and points[each-1] in troughs:
            flag = True
        else:
            flag = False
            
        if flag == True:
            to_delete.append(points[each])
        flag = False
    
    points = [e for e in points if e not in to_delete]
        
    # make dataframe of peaks and troughs separately
    column_names = ['first', 'last']
    heat_df = pd.DataFrame(columns = column_names)
    cool_df = pd.DataFrame(columns = column_names)
    
    if troughs[0] < peaks[0]:
        # this means list starts with a trough (heat event first)
        for each in list(range(len(points)-1)):
            if each % 2 == 0: # if an even index in list
                to_add = [points[each], points[each+1]]
                add_index = len(heat_df)
                heat_df.loc[add_index] = to_add
            elif each %2 != 0: # if an odd index in list
                to_add = [points[each], points[each+1]]
                add_index = len(cool_df)
                cool_df.loc[add_index] = to_add
                
    elif peaks[0] < troughs[0]:
        # list starts with a peak (cool event first)
        for each in list(range(len(points)-1)):
            if each % 2 == 0: # if an even index in list
                to_add = [points[each], points[each+1]]
                add_index = len(cool_df)
                cool_df.loc[add_index] = to_add
            elif each %2 != 0: # if an odd index in list
                to_add = [points[each], points[each+1]]
                add_index = len(heat_df)
                heat_df.loc[add_index] = to_add
                
    plt.figure(3)
    plt.plot(day_top.index, day_top)
    plt.plot(day_top.index[peaks], day_top[peaks], 'x', color='green')
    plt.plot(day_top.index[troughs], day_top[troughs], 'x', color='red')
    plt.title('Heat/Cool events in Top of Tank')
    plt.xticks(rotation=90)
    
    for each in peaks:
        plt.text(day_top.index[each], day_top[each], each)
    for each in troughs:
        plt.text(day_top.index[each], day_top[each], each)
                
    return points, heat_df, cool_df
    

##########################################################################

# Define function to assess the first derivative of samples
# (to be used later by heating and cooling separately)

def get_1st_deriv(event_df):
    # as working with discrete data, can take derivative as chg.y/chg.x
    
    # add an extra 'derivative' column to the df
    event_df['1st Deriv'] = ''
    
    for each in list(range(len(event_df))):
        ch_x = int(event_df.iloc[each,1]) - int(event_df.iloc[each,0])
        ch_y = int(sampled_top[event_df.iloc[each,1]]) - int(sampled_top[event_df.iloc[each,0]])
        deriv = ch_y / ch_x
        event_df.loc[each, '1st Deriv'] = deriv

    return event_df

##########################################################################

# Define function to test goodness of HEATING samples
# is it monotonic?
# is the derivative flat?

# FILTER STEP 1
# want to accept 'good' samples and reject bad ones, including extremes
# want to have a vector of acceptable samples AND all samples for use in 
# report

# Get HEAT stats from functions across whole range

def filter_heat_samples(start,end):
    
    heat_df = get_event_samples(start,end)[1]
    heat_drv_df = get_1st_deriv(heat_df)
     
    # drop n extremes of derivative (top and bottom)
    n = 1
    #heat_drv_df = heat_drv_df.sort_values('1st Deriv').tail(-n).head(-n)
    heat_drv_df = heat_drv_df.sort_values('first')
    
    # trim event samples to avoid 'lip' at start and end
    m = 1
    heat_drv_df['n_first'] = heat_drv_df['first'].add(m)
    heat_drv_df['n_last'] = heat_drv_df['last'].add(-m)
    
    # remove any samples which have length zero
    row_nos = heat_drv_df.index
    
    for each in row_nos:
        if heat_drv_df.loc[each, 'n_last'] - heat_drv_df.loc[each, 'first'] <= 0:
            heat_drv_df = heat_drv_df.drop(each)
    
    # Plot heating events on a histogram to identify a suitable threshold
    plt.figure(5)
    plt.title('Histogram of heating rates')
    plt.hist(heat_drv_df.iloc[:,2], bins=15)
    plt.xlabel('Heating rates, degrees per 30 minute time sample')
    plt.ylabel('Frequency')
    
    # 2 peaks typically appear - a solar heating and an electric heating
    # The smaller peak is the electric heating. Usually around 7/sample
    
    # identify the second peak to use as threshold 
    threshold_rate = 6.5
    
    keep_samples_heat = []
    reject_samples_heat = []
    
    # Check for monotonic increase
    for each in list(range(len(heat_drv_df))):
        # get data for event
        data = sampled_top[heat_drv_df.iloc[each,3]:heat_drv_df.iloc[each,4]+1]
        data_df = pd.DataFrame(data)
        # make new column of change in value
        data_df['Deriv'] = data_df[0].diff()
        
        # is sample monotonic decreasing? Is gradient above a threshold?
        if all(val >= 0 for val in data_df['Deriv'][1:]) and heat_drv_df.iloc[each,2] >= 6.5:
            keep_samples_heat.append(heat_drv_df.index[each])
        else:
            reject_samples_heat.append(heat_drv_df.index[each])
            
    return keep_samples_heat, reject_samples_heat, heat_drv_df


##########################################################################

# Define funciton to test goodness of COOLING samples
# is it monotonic?
# what are the step sizes between successive derivatives?

# FILTER STEP 1
# want to accept 'good' samples and reject bad ones, including extremes
# want to have a vector of acceptable samples AND all samples for use in 
# report

# Get COOL stats from functions across whole range

def filter_cool_samples(start,end):
    cool_df = get_event_samples(start, end)[2]
    cool_drv_df = get_1st_deriv(cool_df)
     
    # drop n extremes of derivative (top and bottom)
    n = 1
    cool_drv_df = cool_drv_df.sort_values('1st Deriv').tail(-n).head(-n)
    cool_drv_df = cool_drv_df.sort_values('first')
    
    # trim event samples to avoid 'lip' at start and end
    m = 3
    cool_drv_df['n_first'] = cool_drv_df['first'].add(m)
    cool_drv_df['n_last'] = cool_drv_df['last'].add(-m)
    
    # remove any samples which have length zero
    row_nos = cool_drv_df.index
    
    for each in row_nos:
        if cool_drv_df.loc[each, 'n_last'] - cool_drv_df.loc[each, 'n_first'] <= 2:
            cool_drv_df = cool_drv_df.drop(each)
    
    keep_samples_cool = []
    reject_samples_cool = []
    
    # Check for monotonic increase
    for each in list(range(len(cool_drv_df))):
        # get data for event
        data = sampled_top[cool_drv_df.iloc[each,3]:cool_drv_df.iloc[each,4]+1]
        data_df = pd.DataFrame(data)
        # make new column of change in value
        data_df['Deriv'] = data_df[0].diff()
        data_df['Drv change'] = abs(data_df['Deriv'].diff())
        
        # is sample monotonic decreasing AND small deriv step changes?
        if all(val <0 for val in data_df['Deriv'][1:]) and all(val < 1.5 for val in data_df['Drv change'][2:]):
            keep_samples_cool.append(cool_drv_df.index[each])
        else:
            reject_samples_cool.append(cool_drv_df.index[each])
            
    # this puts the INDEX LABEL (not positional index) into the keep/reject lists
    
    # observe shapes of keep_samples to check if working
    for each in keep_samples_cool:
        data = sampled_top[cool_drv_df.loc[each, 'n_first']:cool_drv_df.loc[each,'n_last']]
        dates = data.index
        times = [d.strftime('%H:%M') for d in dates]
        data.index = times 
        # plt.figure(6)
        # plt.title('Shapes of Identified Cooling Events')
        # plt.plot(data, 'o')
        # plt.xticks(rotation=90)
        
        date_strings = [d.strftime('%H-%M') for d in dates]
        
    return keep_samples_cool, reject_samples_cool, cool_drv_df

##########################################################################

# Define function to get parameters for HEATING
# include calculation of MAE

def get_heat_rate(keep_samples, heat_drv_df):
    # filter out good samples from dataframe
    samples = heat_drv_df.loc[keep_samples,:]
    heat_rate = sum(samples['1st Deriv'])/len(samples)
    
    return heat_rate

heat_rate = get_heat_rate(filter_heat_samples(0,len(sampled_top)-1)[0], filter_heat_samples(0,len(sampled_top)-1)[2])

##########################################################################

# Define function to get parameters for COOLING
# include calculation of MAE

# FILTER STEP 2
# reject any with MAE over certain value
# return a 'raw' vector for use in report

# get list of samples
keep_samples_cool = filter_cool_samples(0,len(sampled_top)-1)[0]

# get filtered samples
cool_drv_df = filter_cool_samples(0,len(sampled_top)-1)[2]


# get parameters for each of the 'good' samples
        
def log_cool_params2(keep_samples, cool_drv_df):
           
    exp_b = []
    mae = []
    
    for each in keep_samples:
        
        # get data for each sample
        curve_data = sampled_top[cool_drv_df.loc[each,'n_first']:cool_drv_df.loc[each,'n_last']]
        
        # plt.figure(1)
        # plt.plot(curve_data, 'o')
        # plt.xticks(rotation=90)
    
        y = np.array(curve_data, dtype=float) #so the curve_fit can work
        x = np.array(list(range(len(curve_data))), dtype=float)
        
        y_norm = y/y[0]
    
        [popt,pcov] = curve_fit(lambda t,b: np.exp(b*t),  x,  y_norm)
        b = popt[0]
        
        def func(x, b):
            return y[0] * np.exp(b * x)
    
        #popt gives the optimal parameters for a, b
        # plt.figure(10)
        # plt.plot(x, y, 'ko', label="Original Noised Data")
        # plt.plot(x, func(x,b), 'r-', label="Fitted Curve")
        # plt.legend()
        # plt.title('Log curve fit')
        # plt.show()
        
        errors = []
        for each in list(range(len(y))):
            error = y[each] - func(x[each],b)
            errors.append(abs(error))
        
        mean = sum(errors)/len(errors)
        
        exp_b.append(b)
        mae.append(mean)
        
    return exp_b, mae

# Calculate parameters for current samples
[b,mae] = log_cool_params2(keep_samples_cool, cool_drv_df)

data = [keep_samples_cool, b, mae]

params_df = pd.DataFrame(data).transpose()
row_nos = list(range(len(params_df)))
row_nos.reverse()

for each in row_nos:
    if params_df.iloc[each,2] >= 0.1:
        params_df = params_df.drop(each)
        
# Update keep_samples list for final calculation of cooling parameters
use_samples = []

for each in keep_samples_cool:
    if each in list(params_df.iloc[:,0]):
        use_samples.append(each)
        

##########################################################################

# Section to calculate 'actual' parameters using pre-defined functions

# use list 'use_samples' to calculate actual parameters for cooling
use_B = []
use_MAE = []
[use_B, use_MAE] = log_cool_params2(use_samples, cool_drv_df)

plt.figure(50)
plt.title('Sample decay constants vs MAE')
plt.plot(use_B, use_MAE, 'x')
plt.xlabel('Value of decay constant')
plt.ylabel('Value of MAE')
plt.axvline(-0.01786, color='r', linestyle='--')

B = sum(use_B)/len(use_B)

###############################################################################
# export cooling/heating parameters to csv for use in HWT algorithm
model_data = [B, heat_rate]
model_data = pd.DataFrame(model_data)
model_data.to_csv (r'...\model_data.csv', index = False, header=False)

##########################################################################

# At this point, should have acceptable parameters to model HEATING and 
# COOLING 

# Next step is to identify Hot Water usage events

##########################################################################

# Identify USE WATER events within the data

def find_use_water(start, end):
    # use 200 - 300
    
    day_top = sampled_top[start:end]
    day_btm = sampled_btm[start:end]
    
    use_water = []
    
    for each in list(range((end-start)-1)):
        if day_btm[each] - day_btm[each+1] >= 0.5:
            use_water.append(each)
            
    use_water_distanced = [use_water[-1]]
    
    for each in list(range(len(use_water)-1)):
        if abs(use_water[each] - use_water[each+1]) >= 2:
            use_water_distanced.append(use_water[each])

    return start, end, use_water_distanced


# this returns a list of indexes for data points where water has been used

# Collect together according to time of use, show in cumulative histogram
# can then get derivative of this to have a likelihood distribution

# Need to separate out a list of times (HH:MM) associated with sample nos.
# Get indexes for each data point in use_water_distanced

def hw_events(start,end):
    
    water_events = find_use_water(start,end)[2]
    water_times = []
    
    for each in water_events:
        water_times.append(sampled_top.index[each])
    
    # Replace with time in HH:MM format
    water_times = [d.strftime('%H:%M') for d in water_times]
    water_times.sort()
    
    time_list = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range('2022-02-01T00:00:00Z', '2022-02-01T23:30:00Z',
                                      freq='30T'))
       .between_time('00:00','23:30')
       .index.strftime('%H:%M')
       .tolist()
       )
    
    values = [0]*len(time_list)
    
    for each in list(range(len(time_list))):
        number = water_times.count(time_list[each])
        values[each] = number
        
    event_count = pd.Series(values, index=time_list)
    
    plt.figure(8) 
    plt.title('Histogram showing frequency of Hot Water events by time')
    plt.bar(event_count.index, event_count)
    plt.xticks(rotation=90)
    plt.xticks(event_count.index[0::2])
    plt.xlabel('Time, HH:MM')
    plt.ylabel('Frequency')
    
    # now need to create cumulative histogram

    # create cumulative histogram
    no_events = len(water_times)
    event_val = 1/no_events
    y = [0]*len(event_count) # this will be updated with value of each event as it occurs
    
    for each in list(range(len(event_count))):
        y[each] = y[each-1] + (event_val * event_count[each])
        
    plt.figure(9)
    plt.title('Cumulative Relative Histogram for HW events')
    plt.bar(event_count.index, y)
    plt.xticks(rotation=90)
    plt.xticks(event_count.index[0::2])
    plt.xlabel('Time, mm-dd HH')
    plt.ylabel('Relative Cumulative Frequency')
    
    # Can now get LIKELIHOOD DISTRIBUTION as the derivative of this
    data = zip(event_count, y)
    y_df = pd.DataFrame(data, index=event_count.index)
    
    y_df[2] = ''
    
    for each in list(range(1,len(y_df))):
        deriv = y_df.iloc[each, 1] - y_df.iloc[each-1, 1]
        y_df.iloc[each,2] = deriv

    plt.figure(10)
    plt.title('HW Event Likelihood Distribution')
    plt.plot(y_df.index[1:], y_df[2][1:])
    plt.xticks(y_df.index[0::2], rotation=90)
    plt.xlabel('Time, mm-dd HH')
    plt.ylabel('Probablity density')
    
    return y_df
    
    # This y_df distribution will be sampled to predict future day outcomes
    
###############################################################################

# SAMPLE HW EVENTS TO CREATE PREDICTED DEMAND FORECAST

# problem is that we don't know the correct normalisation constant for 
# the pdf

# Use INVERSE TRANSFORM SAMPLING (From C19 topic 4)

# 1. Generate random number z [0,1] from standard uniform distribution
# 2. Compute the value x such that  g(x) = z (g(x) is the cumulative df)
#  ---> x = g-1(x)
# 3. use x to be desired sample

def closest(lst, K):
      
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]

def predict_day(start,end):

    y_df = hw_events(start,end)
    # note columns 0 - frequency, 1 - cumulative, 2 - pdf
    
    z = random.uniform(0,1)
    
    sample_hits = [] # this is a list of 'hits' from sampling
    count = 0
    
    while count < 100:
        z = random.uniform(0,1)
        # considering col 1 of y_df
        # get index of y_df[1] value closest to random no.
        close = closest(list(y_df[1]), z)
        closest_index = y_df.index[y_df[1] == close]
        
        closest_index = closest_index[0]
        sample_hits.append(closest_index)
        count = count + 1

    # add extra column in y_df for counts after sampling
    y_df[3] = [0]*len(y_df) # hits from sampling

    for each in list(range(len(y_df))):
        number = sample_hits.count(y_df.index[each])
        y_df.iloc[each,3] = int(number)
        
    y_df[4] = [each * 100 for each in y_df[2]]
    
    # identify top 5 peaks to have as HW events during the 24 hrs
    y_df = y_df[1:]
    peaks, _ = find_peaks(y_df[3], height=0, distance=4, prominence=0.8)

    y_choose = y_df.iloc[peaks,:]
    y_ch_sort = y_choose.sort_values(3)
    hw_predict = y_ch_sort.index
        
    # print shapes overlaid with pdf
    plt.figure(5)
    plt.title('shape of predicted HWT usage')
    plt.plot(y_df.index, y_df[4])
    plt.plot(y_df.index, y_df[3])
    plt.plot(hw_predict, y_df[3][hw_predict], 'x')
    plt.xticks(y_df.index[0::2], rotation=90)
    
    return hw_predict, y_df

# get predictions of HW events from PDF
y_df = hw_events(0,len(sampled_top)) # pdf is column 2
y_df = y_df[1:]
#peaks, _ = find_peaks(y_df[2], height=0, distance=16, prominence=0)
peaks, _ = find_peaks(y_df[2], height=0, distance=8, prominence=0)


# take top 3 peaks as hw events for the day
peak_values = y_df.iloc[peaks,2]
peak_data = pd.Series(peak_values.values, index=peaks)
peak_data_sorted = peak_data.sort_values(ascending=False)

hw_use = peak_data_sorted[0:3]

plt.figure(10)
plt.title('Chosen Peaks for HW Event')
plt.plot(y_df.index, y_df[2])
plt.plot(y_df.index[hw_use.index], y_df.iloc[hw_use.index,2], 'x')
plt.xticks(y_df.index[0::2], rotation=90)
plt.xlabel('Time, HH:MM')
plt.ylabel('Probability density')

# export to csv
hw_times = y_df.index[hw_use.index]
hw_times = pd.DataFrame(hw_times)

time_list = (pd.DataFrame(columns=['NULL'],
              index=pd.date_range('2022-02-01T00:00:00Z', '2022-02-01T23:30:00Z',
                                  freq='30T'))
   .between_time('00:00','23:30')
   .index.strftime('%H:%M')
   .tolist()
   )

# manipulate hw_times so that it starts one period before 
# get index of start times 
hw_index = []
for each in list(range(2)):
    hw_idx = [idx for idx, element in enumerate(time_list) if element == hw_times.iloc[each,0]]
    hw_index.append(hw_idx[0])

hw_new_idx = [i-1 for i in hw_index]
hw_new_times = []
for each in hw_new_idx:
    hw_new_times.append(time_list[each])

hw_times_output = pd.DataFrame(hw_new_times)
#hw_times_output.to_csv (r'C:\Users\alice\Documents\YR4\4YP\Coding\Optimisation_codes\MM HWT\weekday_times.csv', index = False, header=False)
                      
    
###############################################################################
#------------------------------------------------------------------------------
###############################################################################



    
    
    
    
    
    