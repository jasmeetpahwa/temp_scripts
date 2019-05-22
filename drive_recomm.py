import requests, json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src.models.drive_recommendation.constants import PREDICTION_FILE, API_KEY, BING_LOCATION_API_URL
import time

def populate_geo_coordinates(prediction_file):
    pool_df = pd.read_csv(PREDICTION_FILE)
    result_df = pd.DataFrame(columns=['college', 'count', 'city', 'state', 'lat', 'lng', 'ranking'])
    for index, row in pool_df.iterrows():
        place = row['city'] + ',%20' + row['state']
        response = requests.get(BING_LOCATION_API_URL.format(place, API_KEY))
        x = response.json()
        box = x['resourceSets'][0]['resources'][0]['bbox']
        lat = (box[0] + box[2])/2
        lng = (box[1] + box[3])/2
        row_ = {'ranking':[row['ranking']], 'college':[row['college']], 'count':[row['hirable_students']],\
                'city':[row['city']], 'state':[row['state']], 'lat':[lat], 'lng':[lng]}
        result_df = pd.concat([result_df, pd.DataFrame(row_)])

    result_df.reset_index(drop=True, inplace=True)
    return result_df

def cluster_on_distance(pool_df):

    #distance_threshold of 1.4846059080767864 equals to almost 150 KMs.
    hc = AgglomerativeClustering(affinity = 'euclidean', linkage = 'ward', distance_threshold=1.4846059080767864)
    y_hc = hc.fit_predict(np.array(pool_df[['lat', 'lng']]))
#     for i in range(0, max(y_hc)):
#         plt.scatter(df1[y_hc == i,0], df1[y_hc == i,1], s=100)
#     plt.show()
    pool_df = pd.concat([pool_df, pd.DataFrame(y_hc)], axis=1).rename(columns={0:'cluster_number'})
    pool_df = pool_df.sort_values(by='cluster_number')
    centroid_lat, centroid_lng = [], []
    for i in range(0, pool_df['cluster_number'].nunique()):
        df1 = pool_df[pool_df['cluster_number']==i]
        lt = [df1['lat'].min(), df1['lng'].min()]
        rb = [df1['lat'].max(), df1['lng'].max()]
        centroid_lat.extend([(lt[0] + rb[0])/2] * len(df1))
        centroid_lng.extend([(lt[1] + rb[1])/2] * len(df1))
    pool_df['centriod_lat'] = centroid_lat
    pool_df['centriod_lng'] = centroid_lng
    return pool_df

def cluster_on_count(input_df):
    input_df = input_df.dropna()
    result_df = pd.DataFrame()

    for i in range(input_df['cluster_number'].nunique()):
        df1 = input_df[input_df['cluster_number']==i]
        if not len(df1):
            continue
        max_ = df1['ranking'].max()
        max_count = df1['count'].max()
        norm_count = []
        for index, row in df1.iterrows():
            norm_count.append((row['count']/max_count)*max_)
        df1['norm_count'] = norm_count
        n_clusters = 4 if len(df1) >= 4 else len(df1)
        kmeans = KMeans(n_clusters=n_clusters).fit(df1[['ranking', 'norm_count']])
        df1['internal_clusters'] = list(kmeans.labels_)
        df1.sort_values(by='internal_clusters')
        result_df = pd.concat([result_df, df1])
    result_df = result_df.sort_values(by=['cluster_number', 'internal_clusters'])
    return result_df
 
def make_drives(input_df, company_lat, company_lng, req, no_of_drive_to_recommend=10):
#     def get_latdiff(lat):
#         return np.power(abs(lat-company_lat), 2)
    input_df['lat_diff'] = input_df[['centriod_lat']].apply(lambda x: np.power(abs(x-company_lat), 2))
    
#     def get_lngdiff(lng):
#         return np.power(abs(lng-company_lng), 2)
    input_df['lng_diff'] = input_df[['centriod_lng']].apply(lambda x: np.power(abs(x-company_lng), 2))
    input_df['distance'] = np.sqrt(input_df['lat_diff']+input_df['lng_diff'])
    input_df = input_df.sort_values(by='distance', ascending=True)
    
    def getuniq(l):
        a = []
        i = 0
        while i < len(l):
            s = l[i]
            a.append(s)
            while i < len(l) and l[i] == s:
                i += 1
        return a
            
    x = getuniq(list(input_df['cluster_number']))
    result_df = pd.DataFrame()
    j,i=0,0
    # while i < cluster_to_present and j < len(x):
    while j<len(x):
        temp_df = input_df[input_df['cluster_number']==x[j]]
        i += 1
        j += 1
        group_obj = temp_df.groupby('internal_clusters')['ranking'].apply(np.mean).reset_index()
        group_obj = group_obj.sort_values('ranking')
        a_df = pd.DataFrame()
        sum_ = 0
        for index, row in group_obj.iterrows():
            f_df = temp_df[temp_df['internal_clusters']==row['internal_clusters']]
            for index, row in f_df.iterrows():
    #             display(row)
                a_df = a_df.append(row)
    #             a_df = pd.concat([a_df, row], axis=0)
                sum_ += row['count']
                if sum_>req:
                #     display(a_df)
                #     print(sum_)
                    result_df = pd.concat([result_df, a_df])
                    a_df = pd.DataFrame()
                    sum_ = 0
                    no_of_drive_to_recommend += 1
            if sum_ >= int(0.6*req):
                # display(a_df)
                # print(sum_)
                result_df = pd.concat([result_df, a_df])
                a_df = pd.DataFrame()
                sum_ = 0
                no_of_drive_to_recommend += 1
    return result_df


pool_df_w_geoloc = populate_geo_coordinates(PREDICTION_FILE)
s = time.time()
college_cluster_wrt_distance = cluster_on_distance(pool_df_w_geoloc)
college_cluster_wrt_count_distance = cluster_on_count(college_cluster_wrt_distance)
company_lat = 28.503129
company_lng = 77.083702
req = 1000
result = make_drives(college_cluster_wrt_count_distance, company_lat, company_lng, req)
e = time.time()
print(e-s)
print(result)
# result.to_csv('/home/jasmeet16-jtg/result.csv')

