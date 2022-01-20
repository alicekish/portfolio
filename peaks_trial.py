
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

##########################################################################

# CREATE COST VARIABLE DATA

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
    
randomlist2 = []
for i in range(0,48):
    n = round(random.random(), 2)
    randomlist2.append(n)
    
randomlist3 = []
for i in range(0,48):
    n = round(random.random(), 2)
    randomlist3.append(n)
    
#########################################################################    

# SET UP DEVICE DATAFRAME

# Create template data frames for dev
column_names = ['device_name','power','on_period','use_OFF','s_off','e_off','use_deadline','deadline']
d1_values = ['P2','8','4','false','','','false','']
d2_values = ['P3','4','6','false','','','false','']
d3_values = ['P4','12','9','false','','','false','']

d1_data_raw = [column_names, d1_values]
d2_data_raw = [column_names, d2_values]
d3_data_raw = [column_names, d3_values]

d1_df_raw = pd.DataFrame(d1_data_raw)
d2_df_raw = pd.DataFrame(d2_data_raw)
d3_df_raw = pd.DataFrame(d3_data_raw)

d1_df = d1_df_raw.transpose()
d2_df = d2_df_raw.transpose()
d3_df = d3_df_raw.transpose()

device_data_all = [d1_df.iloc[:,0],d1_df.iloc[:,1], d2_df.iloc[:,1], d3_df.iloc[:,1]]
device_info_df = pd.DataFrame(device_data_all).transpose()
device_info_df.columns = [0,1,2,3]

#########################################################################
# create dataframe of cost variables

Z_data = {'Time':times, 'Z1':randomlist1,'Z2':randomlist2, 'Z3':randomlist3}
Z_df = pd.DataFrame(Z_data)


#######################################################################

# APPLY PEAK AVOIDANCE

# Set a max limit of power
power_limit = 20


# PLACEMENT OF COST BLOCKS
# use For loop to iterate through this list of devices in order, placing
# each at best permissible space

# 1. find best place for largest cost to go. 
# 2. find best place for 2nd largest cost to go
# 3. find best place for 3rd largest cost to go
# 4. find best place for nth largest cost to go


# for each device, ordered by size of Z_min cost
Z1_df = Z_df.iloc[:,[0,1]]
Z1_df_ordered = Z1_df.sort_values('Z1')

Z2_df = Z_df.iloc[:,[0,2]]
Z2_df_ordered = Z2_df.sort_values('Z2')

Z3_df = Z_df.iloc[:,[0,3]]
Z3_df_ordered = Z3_df.sort_values('Z3')


def place_device(device_no, power_list):
    
    # make nZ_df list
    nZ_df = Z_df.iloc[:,[0,device_no]]
    # order nZ_df list
    nZ_df_ordered = nZ_df.sort_values('Z'+str(device_no))
    
    nZ_df_ordered = nZ_df_ordered[nZ_df_ordered.index <= 48 - int(device_info_df.iloc[2,device_no])]
    
    # add a new index to the df
    new_index = list(range(len(nZ_df_ordered)))
    
    nZ_df_ordered['new index'] = new_index

    pwr_list_toadd = [int(device_info_df.iloc[1,device_no])]*int(device_info_df.iloc[2,device_no])
    
    counter = 0
    
    while counter <= len(nZ_df_ordered):
        # for each* in Z_df_ordered list
        # *must make sure that only eligible start times are tested
        # that is, remove start times at end of list that would run over
        # and extend the list beyond length=48

        # get indexes of start and end times for each iteration
        index_start_on = nZ_df_ordered.index[counter]
        index_end_on = index_start_on + int(device_info_df.iloc[2,device_no])-1
        
        # replace into power vector
        power_list[index_start_on:index_end_on+1] = pwr_list_toadd
        
        # check that all values in power vector are below power limit
        if counter >= 48-int(device_info_df.iloc[2,device_no]):
            return([0]*48, 'no time to start device')
            break
        elif all(i <= power_limit for i in power_list) == True:
            return power_list, nZ_df_ordered.iloc[index_start_on,0]
            break
        elif all(i <= power_limit for i in power_list) == False:
            counter = counter + 1
            continue 
        
# now place each device sequentially
power_blank = [0]*48
power_master = np.array(power_blank)

# want to find out which devices have largest => smallest Z costs
x = [Z1_df_ordered.iloc[0,:], Z2_df_ordered.iloc[0,:], Z3_df_ordered.iloc[0,:]]
y = pd.DataFrame(x)

device_nos = list(range(1, len(device_info_df.columns -1)))
values = [y.iloc[0,1], y.iloc[1,2], y.iloc[2,3]]
y['Value'] = values
y['Device_no'] = device_nos

y_sorted = y.sort_values('Value')

# y_sorted is a dataframe with columns 4 having max Z cost and 
# the corresponding device no in column 5

device_ordered_costs = list(y_sorted.iloc[:,5].values)
# this is simply a list of device numbers, ordered by their largest Z cost

for each in device_ordered_costs:
    # this needs to be changed to the ordered list by Z value size?
    
    if place_device(each, list([0]*48))[1] == 'no time to start device':
        print('cannot place device no' + str(each))
        break 
    
    # get the power vector from place_device function for current device
    power_local = np.array(place_device(each, [0]*48)[0])
   
    # add the power vector for device to the master power vector
    power_master = power_master + power_local
    
    # make power limit vector to plot
    pwr_limit = [power_limit]*48
    
    plt.figure(each)
    plt.plot(pwr_limit)
    plt.plot(power_master)
    #plt.plot(power_limit)
    plt.xlabel('1/2 hr period')
    plt.ylabel('Power W')
    plt.xticks(rotation=90)
    plt.title('Power Forecast')
   
    if all(i <= power_limit for i in list(power_master)) == True:
        continue 
    
    elif all(i <= power_limit for i in list(power_master)) == False:
        print('cannot place this device no.' + str(each))
        print('end op.')
        break
    


   
   






