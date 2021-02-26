# turn strings into datetimes
import datetime

data = [['asdf', '2012-01-01 00:00:12', '1234'],
 ['asdf', '2012-01-01 00:00:31', '1235'],
 ['asdf', '2012-01-01 00:00:57', '2345'],
 ['asdf', '2012-01-01 00:01:19', '2346'],
 ['asdf', '2012-01-01 00:01:25', '2345'],
 ['asdf', '2012-01-01 09:04:14', '3465'],
 ['asdf', '2012-01-01 09:04:34', '1613'],
 ['asdf', '2012-01-01 09:04:51', '8636'],
 ['asdf', '2012-01-01 09:05:15', '5847'],
 ['asdf', '2012-01-01 09:05:29', '3672'],
 ['asdf', '2012-01-01 09:05:30', '2367'],
 ['asdf', '2012-01-01 09:05:43', '9544'],
 ['asdf', '2012-01-01 14:48:15', '2572'],
 ['asdf', '2012-01-01 14:48:34', '7483'],
 ['asdf', '2012-01-01 14:48:56', '5782'],
 ['asdf', '2012-01-01 15:48:56', '5782'],
 ['asdf', '2012-01-01 15:50:56', '5782']]

date_format = "%Y-%m-%d %H:%M:%S"

for row in data:
    row[1] = datetime.datetime.strptime(row[1], date_format)
split_dt = datetime.timedelta(minutes=5)
dts = (d1[1]-d0[1] for d0, d1 in zip(data, data[1:]))
split_at = [i for i, dt in enumerate(dts, 1) if dt >= split_dt]
groups = [data[i:j] for i, j in zip([0]+split_at, split_at+[None])]

##################################################################################

import pandas as pd
import datetime    
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

rawdata = [{'filename': 'image_1.jpg',
  'date': datetime.datetime(2014, 11, 13, 19, 14, 16, 152847)},
 {'filename': 'image_2.jpg',
  'date': datetime.datetime(2014, 11, 13, 19, 17, 16, 152847)},
 {'filename': 'image_3.jpg',
  'date': datetime.datetime(2014, 11, 13, 19, 21, 16, 152847)},

 {'filename': 'image_4.jpg',
  'date': datetime.datetime(2014, 11, 13, 20, 20, 16, 152847)},
 {'filename': 'image_5.jpg',
  'date': datetime.datetime(2014, 11, 13, 20, 31, 16, 152847)},
 {'filename': 'image_6.jpg',
  'date': datetime.datetime(2014, 11, 13, 20, 41, 16, 152847)},
 {'filename': 'image_7.jpg',
  'date': datetime.datetime(2014, 11, 13, 20, 51, 16, 152847)},
  
 {'filename': 'image_8.jpg',
  'date': datetime.datetime(2014, 11, 13, 21, 14, 16, 152847)},
 {'filename': 'image_9.jpg',
  'date': datetime.datetime(2014, 11, 13, 21, 17, 16, 152847)},
 {'filename': 'image_10.jpg',
  'date': datetime.datetime(2014, 11, 13, 21, 20, 16, 152847)}]
df = pd.DataFrame(rawdata)


# print(df)
kmeans = KMeans(n_clusters=2)
df['label'] = kmeans.fit_predict(df[['date']])
# print(df)

# ax = df[df['label']==0].plot.scatter(x='date', y='label', s=50, color='white', edgecolor='black')
# df[df['label']==1].plot.scatter(x='date', y='label', s=50, color='white', ax=ax, edgecolor='red')
# plt.scatter(kmeans.cluster_centers_.ravel(), [0.5]*len(kmeans.cluster_centers_), s=100, color='green', marker='*')

# print(df["date"].diff())
# print(df["date"].diff() > pd.Timedelta(minutes=30))
# print((df["date"].diff() > pd.Timedelta(minutes=30)).cumsum())
# df = df.sort_values('date')
# cluster = (df["date"].diff() > pd.Timedelta(minutes=30)).cumsum()
# dfs = [v for k,v in df.groupby(cluster)]
# for clust in dfs:
#     print(clust)

##################################################################################

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

X = [1,2,4,7,9,5,4,7,9,56,57,54,60,200,297,275,243]
X = np.reshape(X, (-1, 1))
# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
# bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)

ms = MeanShift(bandwidth=None, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

# print("number of estimated clusters : %d" % n_clusters_)
# print(labels)

##################################################################################

from numpy import array, linspace
from sklearn.neighbors.kde import KernelDensity
from matplotlib.pyplot import plot

a = array([10,11,9,23,21,11,45,20,11,12]).reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
s = linspace(0,50)
e = kde.score_samples(s.reshape(-1,1))

from scipy.signal import argrelextrema
mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
print ("Minima:", s[mi])
print ("Maxima:", s[ma])
print (a[a < mi[0]], a[(a >= mi[0]) * (a <= mi[1])], a[a >= mi[1]])
plot(s, e)
plt.show()
plot(s[:mi[0]+1], e[:mi[0]+1], 'r',
     s[mi[0]:mi[1]+1], e[mi[0]:mi[1]+1], 'g',
     s[mi[1]:], e[mi[1]:], 'b',
     s[ma], e[ma], 'go',
     s[mi], e[mi], 'ro')
plt.show()
assert(0)

##################################################################################

from scipy.stats import kde
import matplotlib.pyplot as plt
x = sorted([10,11,9,23,21,11,45,20,11,12])

density = kde.gaussian_kde(x) # x: list of price
xgrid = np.linspace(min(x), max(x), 50)   
# plt.plot(xgrid, density(xgrid))
# plt.show()

##################################################################################

np.random.seed(479)
start_date = datetime(2011, 1, 1, 0, 0, 0, 0)
df = pd.DataFrame({ 'date':np.random.choice( 
                    pd.date_range(start_date, periods=365*5, freq='D'), 50) })
df['rel'] = df['date'] - pd.to_datetime(start_date)
df.rel = df.rel.astype('timedelta64[D]')
# df['year_as_float'] = pd.to_datetime(start_date).year + df.rel / 365.
# df['year_as_float'].plot(kind='kde')
# plt.show()

##################################################################################
np.random.seed(0)
dates = pd.date_range('2010-01-01', periods=31, freq='D')
df = pd.DataFrame(np.random.choice(dates,100), columns=['dates'])
# use toordinal() to get datenum
df['ordinal'] = [x.toordinal() for x in df.dates]
print(df)

# plot non-parametric kde on numeric datenum
ax = df['ordinal'].plot(kind='kde')
# rename the xticks with labels
x_ticks = ax.get_xticks()
ax.set_xticks(x_ticks[::2])
xlabels = [datetime.fromordinal(int(x)).strftime('%Y-%m-%d') for x in x_ticks[::2]]
ax.set_xticklabels(xlabels)
# plt.show()

s = pd.Series(['25/01/2000 05:50', '25/01/2000 05:50', '25/01/2000 05:50']) 
s = pd.to_datetime(s) # make sure you're dealing with datetime instances 
s.apply(lambda v: v.timestamp()) 
a = datetime.strptime('2019-12-27 11:24:48', '%Y-%m-%d %H:%M:%S').timestamp() 
import time
timestamp = 1366831506000
print(time.strftime("%a %d %b %Y %H:%M:%S GMT", time.gmtime(timestamp / 1000.0)))