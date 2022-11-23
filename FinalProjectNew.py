from unicodedata import category
import matplotlib.pyplot as plt
import os
import sqlite3
import unittest
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
import numpy as np
from pathlib import Path

from function_storage import AutoRegModel
#from function_storage import top, get_stats, AutoRegModel, get_wavg
from function_storage import helperset



def get_restaurant_data(db_filename):

    conn = sqlite3.connect('South_U_Restaurants.db')
    cur = conn.cursor()
    cur.execute(
        """
        SELECT restaurants.name, categories.category, buildings.building, restaurants.rating
        FROM restaurants JOIN categories
        ON restaurants.category_id = categories.id
        JOIN buildings
        ON restaurants.building_id = buildings.id
        """
    )
    conn.commit()

    data = cur.fetchall()
    #create an empty list
    my_list = [] 
    #print(data)
        #data prints out a list of tuples that contain all the information provided by the three data tables
        #looping through for resturant in data
    for r in data:
        #create a new dictionary 
        my_dict = {}
        #create four items in the dictionary and set them equal to the values provided by the tuples in the list items of data
        my_dict['name'] = r[0]
        my_dict['category'] = r[1]
        my_dict['building'] = r[2]
        my_dict['rating'] = r[3]
        #append this new dictionary to the list we created 
        my_list.append(my_dict)
    return my_list
#running first function
#get_restaurant_data('South_U_Restaurants.db')


def charts_restaurant_categories(db_filename):
    restaurant_list = get_restaurant_data(db_filename)
    my_dict = {}
    for r in restaurant_list:
        #if r['category'] in my_dict:
            #my_dict[r['category']] += 1
        #my_dict[r['category']] = 1
        my_dict[r['category']] = my_dict.get(r['category'], 0) + 1

    #print(my_dict)

    category_list = []
    count_list = []
    for c in my_dict:
        category_list.append(c)
        count_list.append(my_dict[c])
    plt.barh(category_list, count_list, color = 'blue')
    plt.xlabel('Number of restaurants')
    plt.ylabel('restaurant category')
    plt.title('Types of restaurant on South University Ave')
    plt.show()

    plt.pie(count_list, labels = category_list)
    plt.title('Types of restaurant on South University Ave')
    plt.show()
    
    return my_dict

#charts_restaurant_categories('South_U_Restaurants.db')


#all functions in this section come from function_storage file 
#helping modularize the project



restaurant_data_df = pd.DataFrame(get_restaurant_data("South_U_Restaurants.db"))

restaurant_data_dfTwo = restaurant_data_df.copy()

restaurant_data_dfTwo['category_int'] = enc.fit_transform(restaurant_data_dfTwo[['category']])

restaurant_data_dfTwo['category_int'] = restaurant_data_dfTwo['category_int'].astype(int)

AutoRegModel(restaurant_data_dfTwo, 5)

#UPDATE: using the helperset class and saving it as the variable data to call on

data =  helperset(restaurant_data_dfTwo)  

print(data.top(6))

#grouped = restaurant_data_df.groupby(["name"])
grouped = restaurant_data_dfTwo.groupby(level = 0).sum()
#can apply the function instead of calling on it... 
print(data.get_stats(grouped), data.get_wavg(grouped))

#prints out a massive list of the data stored in the database South_U_Restaurants
#each item in this list is a dictionary 
#print(get_restaurant_data('South_U_Restaurants.db'))
# Creating dataset


def threeDPlot(data):
        z = data.rating
        x = data.building
        y = data.category_int
 
# Creating figure
        fig = plt.figure(figsize = (10, 8))
        ax = plt.axes(projection ="3d")
   
# Add x, y gridlines
        ax.grid(b = True, color ='black',
                linestyle ='--', linewidth = 0.3,
                )
 
# Creating color map
        my_cmap = plt.get_cmap('coolwarm')
 
# Creating plot
        sctt = ax.scatter3D(x, y, z,
                #    alpha = .9,
                    c = z,
                    cmap = my_cmap,
                    marker ='*')
 
#plt.title(' | '.join(restaurant_data_dfname))
        ax.set_xlabel('building', fontweight ='bold')
        ax.set_ylabel('category_int', fontweight ='bold')
        ax.set_zlabel('rating', fontweight ='bold')
        fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
 
# show plot
        plt.show()

#threeDPlot(restaurant_data_dfTwo)
              
from sklearn.preprocessing import OrdinalEncoder
def final_plots():
    
    enc = OrdinalEncoder()
    restaurant_data_df['category_int'] = enc.fit_transform(restaurant_data_df[['category']])
    restaurant_data_df['category_int'] = restaurant_data_df[['category_int']].astype(int)

    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 7.5})
    sns.pairplot(data=restaurant_data_df);


    cor = round(restaurant_data_df.corr(), 5)
    cor
    plt.figure(figsize=(6,5))
    sns.heatmap(cor, annot= True, cmap=plt.cm.CMRmap_r)
    plt.title('Heatmap for the Data Correlation')
    plt.show()

    restaurant_data_df.hvplot.scatter('category_int', "rating")
    restaurant_data_df.hvplot.barh('category_int', "rating")

final_plots()