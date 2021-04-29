import pandas as pd
from numba import vectorize, float64, numba
from skmob.utils import plot
import folium
from skmob.measures.individual import *
import skmob
from skmob.utils.gislib import getDistanceByHaversine

import warnings
warnings.filterwarnings("ignore")

def save_csv_zipped(df, name, folder):

	file_path = '../data/'+folder+"/"+name

	compression_opts = dict(method='zip', archive_name=name+'.csv')  

	df.to_csv(file_path+'.zip', sep='\t', index=False, compression=compression_opts)



def plot_points(coords, map_f=None):

    lats, lngs = zip(*coords)
    if map_f is None:
        m = folium.Map([np.mean(lats),np.mean(lngs)], zoom_start=10, tiles='cartodbpositron')
    else:
        m = map_f
    for coord in coords:
        folium.Circle(location=[coord[0], coord[1]], fill_color='#ffffff', radius=8).add_to(m)
    return m



@vectorize([float64(float64, float64, float64, float64)])
def vect_dist(lat0,lng0,lat1,lng1):
    return getDistanceByHaversine((lat0,lng0),(lat1, lng1))



def report_results_cv(df_result, n_top=10):
    
    df_rank = df_result.sort_values(['rank_test_score'])[:n_top][['rank_test_score','params',
                                                    'mean_test_score','std_test_score']]
    
    for rank, params, avg_score, std in zip(df_rank['rank_test_score'],df_rank['params'],
                                           df_rank['mean_test_score'],
                                            df_rank['std_test_score']):
        
            print("Model with rank: {0}".format(rank))
            print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
                  avg_score, std))
            print("Parameters: {0}".format(params))
            print("")




def print_association_rules(ar_list):
    
    for item in ar_list:
        # first index of the inner list
        # Contains base item and add item
        # the rules are items_base -> items_add
        items_base = [x for x in item[2][0][0]]
        items_add = [x for x in item[2][0][1]]
        print("Rule: " + str(items_base) + " -> " + str(items_add))

        #second index of the inner list
        print("Support: " + str(item[1]))

        #third index of the list located at 0th
        #of the third index of the inner list

        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")