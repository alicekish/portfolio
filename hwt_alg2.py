#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:58:28 2022

@author: alice
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import random
import requests
import math


# HWT ALGORITHM
# have times when HW is needed
# have models for heating and cooling

# RUN between 12:55 and 1AM
# get times for HW events from file
# get current temperature of top of tank from file via Node Red
# model cooling up until first HW event
# determine when best to switch on for that event
# model heating and subsequent cooling until next event
# continue for all predicted events

# use real or trial mode
use_real_hwt_data = False
use_real_forecasts = False

##############################################################################

##############################################################################
# get current temperature of HW tank. Get temp at 00:55AM
# Set this as 1AM time

# need to get data from Node Red via Pi
if use_real_hwt_data == True: 
    morning_data = pd.read_csv(r'/home/pi/Documents/MM_HWT/morning_values.csv', header=None)
    data = pd.DataFrame(morning_data).transpose()
    y0 = data.iloc[2,0]
    y0_1am = float(y0[9:])
    hw_times = pd.read_csv(r'/home/pi/Documents/MM_HWT/hw_times.csv', header=None)
    model_data = pd.read_csv(r'/home/pi/Documents/MM_HWT/model_data.csv', header=None)

    
elif use_real_hwt_data == False:
    #make a 1AM temperature up
    y0_1am = random.uniform(20,30)
    model_data = pd.read_csv(r'...\HWT TESTING\model_data.csv', header=None)
    hw_times = pd.read_csv(r'...\HWT TESTING\hw_times.csv', header=None)


##############################################################################
# Define function to model cooling from point to point

# note this is in form, B, heat_rate
B = model_data.iloc[0,0]


# generate a general dataframe to hold times and values from start to end 
temp_df = pd.DataFrame()   
temp_df[0] = list(range(48))

time_list = (pd.DataFrame(columns=['NULL'],
              index=pd.date_range('2022-02-01T01:00:00Z', '2022-02-02T00:30:00Z',
                                  freq='30T'))
   .between_time('01:00','00:30')
   .index.strftime('%H:%M')
   .tolist()
   )

temp_df[1] = time_list
temp_df['temp_predict'] = [0]*len(temp_df)

def model_cool(start,end, y0):

    def func(x, b, y0):
        return y0 * np.exp(b * x)
    
    # make empty data frame
    cool_df = pd.DataFrame()   
    cool_df[0] = list(range(48))

    time_list = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range('2022-02-01T01:00:00Z', '2022-02-02T00:30:00Z',
                                      freq='30T'))
       .between_time('01:00','00:30')
       .index.strftime('%H:%M')
       .tolist()
       )

    cool_df[1] = time_list
    
    # get equivalent indexes for start and end times
    start_index = cool_df.index[cool_df[1] == start]
    end_index = cool_df.index[cool_df[1] == end]
    end_index = end_index +1
    
    # trim data list down
    data = cool_df[start_index[0]:end_index[0]]
    
    values = []  
    for each in list(range(len(data))):
         values.append(func(each, B, y0))
    
    data.loc[:,2] = values
    
    return data
    
#-----------------------------------------------------------------------------
# Define function to model heating
# Define heat rate to model heating
heat_rate = model_data.iloc[1,0]

def model_heat(start_temp, heat_length):
    
    heat_temps1 = [start_temp]
    each_temp = start_temp
    for each in list(range(heat_length)):
        each_temp = each_temp + heat_rate
        if each_temp > 65:
            each_temp = 65
        heat_temps1.append(each_temp)
    return heat_temps1

###############################################################################
# DEFINE FUNCTIONS TO GENERATE TEMPLATE CI AND PRICE FORECASTS

# generate a dataframe of times and CI values 
# parameters: length --> length of dataframes required, usually 48

def make_ci(length):
    # create list of times
    times = []
    for hour in range(24):
        for minute in range(0, 60, 30):
            times.append('{:02d}:{:02d}'.format(hour, minute))
            
    # Create list of optimisation variable
    import random
    randomlist1 = []
    for i in range(0,48):
        n = round(random.random(), 2)
        randomlist1.append(n)
    
    ci_data = {'Time':times[2:length], 'CI':randomlist1[2:length]}
    ci_df= pd.DataFrame(ci_data)
    
    return ci_df
        
def make_price(length):
    # create list of times
    times = []
    for hour in range(24):
        for minute in range(0, 60, 30):
            times.append('{:02d}:{:02d}'.format(hour, minute))
            
    # Create list of optimisation variable
    import random
    randomlist1 = []
    for i in range(0,48):
        n = round(random.random(), 2)
        randomlist1.append(n)
    
    price_data = {'Time':times[2:length], 'Price':randomlist1[2:length]}
    price_df= pd.DataFrame(price_data)
    
    return price_df        
    
#------------------------------------------------------------------------------    
# DEFINE FUNCTION TO GET REAL CI DATA FROM API

def get_ci_data():
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    co2_base_url = "https://api.carbonintensity.org.uk"
    co2_url = (co2_base_url+"/intensity/"+now+"/fw24h")
    co2 = requests.get(co2_url)
    
    intensity = co2.json()
    
    ci_df = pd.json_normalize(intensity, 'data')
    ci_df = ci_df[['from','intensity.forecast']]
    
    ci_df.iloc[:,0] = [w[11:16] for w in ci_df.iloc[:,0]]
    ci_df = ci_df.iloc[1:,:]
    ci_df = ci_df.reset_index(drop=True)
    
    return ci_df

def get_price_data():
    
    now = datetime.datetime.today()
    now_fw24 = datetime.datetime.today() + datetime.timedelta(days=1)

    now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    now_fw24_str = now_fw24.strftime("%Y-%m-%dT%H:%M:%SZ")

    url = ('https://api.octopus.energy/v1/products/AGILE-18-02-21/' + 
             'electricity-tariffs/E-1R-AGILE-18-02-21-C/standard-unit-rates/' + 
             '?period_from=' + now_str +'&period_to=' + now_fw24_str)
    r = requests.get(url)
    output_dict = r.json()

    valid_from = [x['valid_from'] for x in output_dict['results']]
    value_exc_vat = [x['value_exc_vat'] for x in output_dict['results']]
    
    prices_data = [valid_from, value_exc_vat]

    prices_df_raw = pd.DataFrame(prices_data)
    
    prices_df = prices_df_raw.transpose()
    
    prices_df.iloc[:,0] = [w[11:16] for w in prices_df.iloc[:,0]]
    
    prices_df = prices_df.iloc[:-1,:]
    
    prices_reset = prices_df.reindex(index=prices_df.index[::-1])
    prices_df = prices_reset.reset_index(drop=True)

    
    return prices_df

   
#------------------------------------------------------------------------------
# DEFINE FUNCTION TO MAKE PRICE WEIGHTING

def make_pr_weight(weight):
    price_weight = weight
    return price_weight

#------------------------------------------------------------------------------
# DEFINE FUNCTION TO GET PRICE WEIGHTING FROM NODE RED    

def get_pr_weight():
    price_weight = pd.read_csv(r'/home/pi/Documents/Working/price_weighting.csv', header=None)
    price_weight = price_weight[0][0]
    return price_weight
    
#------------------------------------------------------------------------------
if use_real_forecasts == True:
    ci_df = get_ci_data()
    pr_df = get_price_data()
elif use_real_forecasts == False:
    ci_df = make_ci(48)
    pr_df = make_price(48)
    
if use_real_hwt_data == True:
    pr_weight = get_pr_weight()
elif use_real_hwt_data == False:
    pr_weight = make_pr_weight(0.5)
    
###############################################################################

# want to identify no. of samples it takes to cool from Tmax to Tthresh
# this gives EARLIEST time that we can finish heating before HW event

# set Tmax and Tthresh
T_max = 65
T_th = 50

# set length of HW event (ie time to continuously heat whilst hot water used)
# note this is in no of 30 minute samples
hw_length = 3

# cooling it T = T0exp(bt) so t = ln(Tthresh/Tmax)/b
samples_max2thr = np.log(T_th/T_max)/B # this is in no. of 30 min samples

# round to whole sample no.
samples_max = math.floor(samples_max2thr)

# SO options for heating are within this time period of *roughly* 14 samples

# Identify where in sample list heating events occur
hw_timelist = list(hw_times.iloc[:,0])

hw_event_samples = []
for each in list(range(len(hw_timelist))):
    event_index = temp_df.index[temp_df[1] == hw_timelist[each]]
    hw_event_samples.append(event_index)

###############################################################################
# Process forecasts to produce a costs (Z_df) dataframe

# NORMALISE DATA

# make CI and Price dataframes the same length
time_length = min(len(ci_df), len(pr_df))
ci_df_crop = ci_df.iloc[:time_length, :]
pr_df_crop = pr_df.iloc[:time_length,:]

# normalise values
norm_ci = ci_df_crop.iloc[:,1]/ci_df_crop.iloc[:,1].abs().max()
norm_pr = pr_df_crop.iloc[:,1]/pr_df_crop.iloc[:,1].abs().max()

# replace into CI and Price cropped dfs
ci_df_crop.iloc[:,1] = norm_ci
pr_df_crop.iloc[:,1] = norm_pr

#----------------------------------------------------------------------------------------------------
# APPLY WEIGHTINGS AND CREATE COST VARIABLE DF

# get price and ci weightings
p = pr_weight
c = 1 - pr_weight

# make new dataframe for comparison
compare_df = pd.DataFrame(zip(ci_df_crop.iloc[:,0], ci_df_crop.iloc[:,1], pr_df_crop.iloc[:,1]), columns=["Time", "CI", "Price"])

# apply optimisation 
z1 = c*compare_df['CI'] # 1st column is CI
z2 = p*compare_df['Price'] # 2nd column is Price

Z = z1.add(z2)

Z_data = [compare_df["Time"], Z]
Z_df = pd.DataFrame(Z_data).transpose()
# this Z_df is a dataframe of costs across the next 24 hours which will then be
# used for each heating event, cropped to the available heating times


###############################################################################
# TO FIND BEST HEATING OPTION FOR HEATING EVENT
# for each start heating option in temp_df_n
    # want to calculate integrated cost for each time according to ON time
    # for heating
    # add this into new column

# temp0 is the starting temperature (this can be found from the previous
# dataframe)

def optimise_heating(temp_df_n, temp0, hw_event_no):
    
    # populate 'temp_predict' colum
    cool_data_n = model_cool(temp_df_n.iloc[0,1], temp_df_n.iloc[-1,1], temp0)
    temp_df_n.loc[:,'temp_predict'] = cool_data_n.iloc[:,2]
    temp_df_n = temp_df_n[0:-1]
    
    # add column that gives time to heat to Tmax
    time_toheat = []
    for each in list(range(len(temp_df_n))):
        time = math.ceil((T_max - temp_df_n.iloc[each,2])/heat_rate)
        time_toheat.append(time)
    
    temp_df_n.loc[:,'time_toheat'] = [0]*len(temp_df_n)
    temp_df_n.iloc[:,3] = time_toheat
    
    # trim available times to check they are within the max sample range for cooling
    if samples_max <= len(temp_df_n):
        temp_df_n = temp_df_n.iloc[len(temp_df_n)-samples_max:,:]
    else:
        pass
    
    temp_heated = []
    for each in list(range(len(temp_df_n))):
        temp = temp_df_n.iloc[each,2] + (heat_rate*temp_df_n.iloc[each,3])
        temp_heated.append(temp)
    
    temp_df_n.loc[:,'temp_heated'] = temp_heated
    
    # round values down to max heat as boiler automatically shuts off at Tmax
    temp_ceiling = []
    for each in list(range(len(temp_df_n))):
        temp_ceil = min(T_max, temp_df_n.iloc[each,4])
        temp_ceiling.append(temp_ceil)
    
    temp_df_n.loc[:,'temp_heat_round'] = temp_ceiling
    
    temp_df_n = temp_df_n.rename({1:'start_heat'}, axis=1)
    
    # get times after heating
    t_postheat = []
    for each in list(range(len(temp_df_n))):
        time_index = time_list.index(temp_df_n.iloc[each,1])
        index_postheat = time_index + temp_df_n.iloc[each,3]
        time_postheat = time_list[index_postheat]
        t_postheat.append(time_postheat)
    
    temp_df_n.loc[:,'time_postheat'] = t_postheat
    
    # trim costs Z_df and temp_df_n to fit available options from temp_df_n    
    start_index = temp_df_n.index[0]

    # get time when time_postheat is equal to the target hw time
    # if time not available, get next earlier time    
    if hw_times.iloc[hw_event_no-1,0] in list(temp_df_n.iloc[:,6]):
        end_index = temp_df_n.index[temp_df_n.loc[:,'time_postheat'] == hw_times.iloc[hw_event_no-1,0]]
        end_index = int(end_index.values)
    elif hw_times.iloc[hw_event_no-1,0] not in list(temp_df_n.iloc[:,6]):
        # get list of times
        time_index = temp_df.index[temp_df[1] == hw_times.iloc[hw_event_no-1,0]]
        earlier_time = temp_df.iloc[time_index-1, 1]
        end_index = temp_df_n.index[temp_df_n.loc[:,'time_postheat'] == str(earlier_time.values)[2:-2]]
        end_index = int(end_index.values)

    
    temp_df_n = temp_df_n.loc[start_index:end_index,:]
    Z_df_crop = Z_df.loc[start_index:end_index,:]
    a = list(range(len(Z_df_crop)))
    counter = 0
    sum_results = []
    
    for each in a:
        on_period = temp_df_n.iloc[each, 3]
        # create list of values to sum
        position = a[counter]
        to_sum = Z_df_crop.iloc[int(position):int(position+on_period), 1]
        #sum each possible scenario and save to results vector
        sum_results.append(sum(to_sum))
        counter = counter + 1
    
    results = [Z_df_crop.iloc[:,0].values, sum_results]
    results_df = pd.DataFrame(results).transpose()                                  
    
    # find minimum time                
    y_min = results_df[1].min()
    min_time_index = results_df.index[results_df.iloc[:,1] == y_min][0]
    #min_time_index = temp_df_n.index[results_df.iloc[:,1] == y_min]
    min_time = results_df.iloc[min_time_index,0]
    
    return min_time, min_time_index, temp_df_n

###############################################################################
# ON and OFF instructions need to include hw_length ON during HW usage
hw_on = list(hw_times.iloc[:,0])
hw_off_indx = [time_list.index(i)+hw_length for i in hw_on]
hw_off = list([time_list[i] for i in hw_off_indx])

# Make list of ON/OFF times for csv output
ON = hw_on
OFF = hw_off


###############################################################################
# HW EVENT 0-1

# Produce temp_df_1 which gives data for available heating times 
# get predicted temperatures for all times from 01AM to first hw event 

temp_df_1 = temp_df.iloc[0:hw_event_samples[0][0]+1+hw_length,:]
start_cool1 = temp_df_1.iloc[0,1]
end_cool1 = temp_df_1.iloc[-1,1]
y01 = y0_1am
temp_df_1.loc[:,'temp_predict'] = model_cool(start_cool1, end_cool1, y01)[2]

if temp_df_1.iloc[-1,2] >= T_th:
    # no cooling needed. Can leave as just cooling
    temp_df_1.iloc[-hw_length:,2] = temp_df_1.iloc[-hw_length,2]
    pass
elif temp_df_1.iloc[-1,2] < T_th:
    # heating required. 
    temp_df1_working = optimise_heating(temp_df_1, temp_df_1.iloc[0,2], 1)[2]
    opt_time1 = optimise_heating(temp_df_1, temp_df_1.iloc[0,2],1)[0]
    opt_time1_idx = optimise_heating(temp_df_1, temp_df_1.iloc[0,2],1)[1]
    opt_time1_idx = temp_df_1[0][temp_df_1.iloc[:,1] == str(opt_time1)]
    
    # get temperature at start of 1st heating event
    opt_time1_idx = str(opt_time1_idx.values)[1:-1]
    opt_time1_idx = int(opt_time1_idx)
    
    # update temp_df_1 with heating/cooling values
    get_start_temp1 = temp_df_1.loc[opt_time1_idx,'temp_predict']
    time_heat1 = temp_df1_working.loc[opt_time1_idx,'time_toheat']

    # add these values to the heating section of temp_df_1
    temp_df_1.loc[opt_time1_idx:opt_time1_idx+time_heat1,'temp_predict'] = model_heat(get_start_temp1, time_heat1)

    # now model cooling for rest of temp_df_2 up until heat event
    start_cool2 = temp_df_1.loc[opt_time1_idx+time_heat1,1]
    end_cool2 = temp_df_1.iloc[-(hw_length+1),1]
    start_cool2_temp = temp_df_1.loc[opt_time1_idx+time_heat1,'temp_predict']

    cool_temps2 = model_cool(start_cool2, end_cool2, start_cool2_temp)
    temp_df_1.loc[opt_time1_idx+time_heat1:,'temp_predict'] = cool_temps2[2]
    
    # hold temperature at this last value whilst hw is used and heating is ON
    temp_df_1.iloc[-hw_length:,2] = cool_temps2.iloc[-1,2]

    # Update ON/OFF lists
    ON.append(str(opt_time1))
    OFF.append(temp_df_1.loc[opt_time1_idx + time_heat1,1])
    
    
plt.figure(0)
plt.title('temp_df_1') 
plt.plot(temp_df_1.iloc[:,1], temp_df_1.iloc[:,2])
plt.xticks(temp_df_1.iloc[:,1][0::2], rotation=90)   
    
    
###############################################################################
# HW EVENT 1-2
temp_df_2 = temp_df.iloc[hw_event_samples[0][0]+hw_length:hw_event_samples[1][0]+1 + hw_length,:]

start_cool2 = temp_df_2.iloc[0,1]
end_cool2 = temp_df_2.iloc[-1,1]
y02 = temp_df_1.iloc[-1,2]
temp_df_2.loc[:,'temp_predict'] = model_cool(start_cool2, end_cool2, y02)[2]

if temp_df_2.iloc[-1,2] >= T_th:
    # no cooling needed. Can leave as just cooling
    temp_df_2.iloc[-hw_length:,2] = temp_df_2.iloc[-hw_length,2]
    pass
elif temp_df_2.iloc[-1,2] < T_th:
    # heating required.  
    temp_df2_working = optimise_heating(temp_df_2, temp_df_2.iloc[0,2], 2)[2]
    opt_time2 = optimise_heating(temp_df_2, temp_df_2.iloc[0,2],2)[0]
    opt_time2_idx = optimise_heating(temp_df_2, temp_df_2.iloc[0,2],2)[1]
    opt_time2_idx = temp_df_2[0][temp_df_2.iloc[:,1] == str(opt_time2)]

    # get temperature at start of 1st heating event
    opt_time2_idx = str(opt_time2_idx.values)[1:-1]
    opt_time2_idx = int(opt_time2_idx)
    
    # update temp_df_1 with heating/cooling values
    get_start_temp2 = temp_df_2.loc[opt_time2_idx,'temp_predict']
    time_heat2 = temp_df2_working.loc[opt_time2_idx,'time_toheat']

    # add these values to the heating section of temp_df_1
    temp_df_2.loc[opt_time2_idx:opt_time2_idx+time_heat2,'temp_predict'] = model_heat(get_start_temp2, time_heat2)

    # now model cooling for rest of temp_df_2
    start_cool3 = temp_df_2.loc[opt_time2_idx+time_heat2,1]
    end_cool3 = temp_df_2.iloc[-(hw_length+1),1]
    start_cool3_temp = temp_df_2.loc[opt_time2_idx+time_heat2,'temp_predict']

    cool_temps3 = model_cool(start_cool3, end_cool3, start_cool3_temp)
    temp_df_2.loc[opt_time2_idx+time_heat2:,'temp_predict'] = cool_temps3[2]

    # hold temperature at this last value whilst hw is used and heating is ON
    temp_df_2.iloc[-hw_length:,2] = cool_temps3.iloc[-1,2]

    # Update ON/OFF lists
    ON.append(str(opt_time2))
    OFF.append(temp_df_2.loc[opt_time2_idx + time_heat2,1])
    
plt.figure(1)
plt.title('temp_df_2') 
plt.plot(temp_df_2.iloc[:,1], temp_df_2.iloc[:,2])
plt.xticks(temp_df_2.iloc[:,1][0::2], rotation=90)   
    
###############################################################################
# HW EVENT 2-3
temp_df_3 = temp_df.iloc[hw_event_samples[1][0]+hw_length:hw_event_samples[2][0]+1 + hw_length,:]

start_cool3 = temp_df_3.iloc[0,1]
end_cool3 = temp_df_3.iloc[-1,1]
y03 = temp_df_2.iloc[-1,2]
temp_df_3.loc[:,'temp_predict'] = model_cool(start_cool3, end_cool3, y03)[2]

if temp_df_3.iloc[-1,2] >= T_th:
    # no cooling needed. Can leave as just cooling
    temp_df_3.iloc[-hw_length:,2] = temp_df_3.iloc[-hw_length,2]
    pass
elif temp_df_3.iloc[-1,2] < T_th:
    # heating required. 
    temp_df3_working = optimise_heating(temp_df_3, temp_df_3.iloc[0,2], 3)[2]
    opt_time3 = optimise_heating(temp_df_3, temp_df_3.iloc[0,2],3)[0]
    opt_time3_idx = optimise_heating(temp_df_3, temp_df_3.iloc[0,2],3)[1]
    opt_time3_idx = temp_df_3[0][temp_df_3.iloc[:,1] == str(opt_time3)]
    
    # get temperature at start of 1st heating event
    opt_time3_idx = str(opt_time3_idx.values)[1:-1]
    opt_time3_idx = int(opt_time3_idx)
    
    # update temp_df_1 with heating/cooling values
    get_start_temp3 = temp_df_3.loc[opt_time3_idx,'temp_predict']
    time_heat3 = temp_df3_working.loc[opt_time3_idx,'time_toheat']

    # add these values to the heating section of temp_df_1
    temp_df_3.loc[opt_time3_idx:opt_time3_idx+time_heat3,'temp_predict'] = model_heat(get_start_temp3, time_heat3)

    # now model cooling for rest of temp_df_2
    start_cool4 = temp_df_3.loc[opt_time3_idx+time_heat3,1]
    end_cool4 = temp_df_3.iloc[-(hw_length+1),1]
    start_cool4_temp = temp_df_3.loc[opt_time3_idx+time_heat3,'temp_predict']

    cool_temps4 = model_cool(start_cool4, end_cool4, start_cool4_temp)
    temp_df_3.loc[opt_time3_idx+time_heat3:,'temp_predict'] = cool_temps4[2]

    # hold temperature at this last value whilst hw is used and heating is ON
    temp_df_3.iloc[-hw_length:,2] = cool_temps4.iloc[-1,2]

    # Update ON/OFF lists
    ON.append(str(opt_time3))
    OFF.append(temp_df_3.loc[opt_time3_idx + time_heat3,1])
    
plt.figure(2)
plt.title('temp_df_3') 
plt.plot(temp_df_3.iloc[:,1], temp_df_3.iloc[:,2])
plt.xticks(temp_df_3.iloc[:,1][0::2], rotation=90)   
    
###############################################################################
# model cooling for after final HW event
temp_df_4 = temp_df.iloc[hw_event_samples[2][0]+hw_length:,:]
start_cool4 = temp_df_4.iloc[0,1]
end_cool4 = temp_df_4.iloc[-1,1]
y04 = temp_df_3.iloc[-1,2]
temp_df_4.loc[:,'temp_predict'] = model_cool(start_cool4, end_cool4, y04)[2]

# append all predicted temperatures together
temps1 = list(temp_df_1.iloc[:,2].values)
temps2 = list(temp_df_2.iloc[1:,2].values)
temps3 = list(temp_df_3.iloc[1:,2].values)
temps4 = list(temp_df_4.iloc[1:,2].values)

temps = temps1 + temps2 + temps3 + temps4

temp_df.iloc[:,2] = temps

plt.figure(4)
plt.title('Predicted Top Temperature Forecast')
plt.plot(temp_df.iloc[:,1], temp_df.iloc[:,2])
plt.axhline(y=T_max, color='r', linestyle='--')
plt.axhline(y=T_th, color='g', linestyle='--')
plt.axvline(x = hw_times.iloc[0,0], color='r')
plt.axvline(x = hw_times.iloc[1,0], color='r')
plt.axvline(x = hw_times.iloc[2,0], color='r')
plt.xticks(temp_df.iloc[:,1][0::2], rotation=90)
plt.xlabel('Time HH:MM')
plt.ylabel('Top Temperature, degrees C')

# remove any duplicates in OFF list to avoid sending conflicting signals to tank
ON_final = ON
OFF_final = [i for i in OFF if i not in ON]

###############################################################################
# export ON and OFF lists to csv                          
control_df = pd.DataFrame([ON,OFF]).transpose()
control_df.columns = ['ON_final','OFF_final']

control_df.to_csv(r'/home/pi/Documents/MM_HWT/HWT_control.csv')





















