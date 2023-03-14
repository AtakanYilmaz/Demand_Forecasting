
###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Kural tabanlı müşteri segmentasyonu
# yöntemi RFM ile makine öğrenmesi yöntemi
# olan K-Means'in müşteri segmentasyonu için
# karşılaştırılması beklenmektedir.

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Değişkenler
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

import datetime as dt
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.neighbors import LocalOutlierFactor
from helpers.eda import *
from helpers.data_prep import *



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("Hafta_03/Ders Öncesi Notlar/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.shape



###############################################################
# 3. Veri Hazırlama (Data Preparation)
###############################################################


df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.shape

# RFM Analizinde kullanım alanı olmadığı için StockCode,Description ve Country değişkenlerinden kurtulalım.

df = df.drop(['StockCode', 'Description', 'Country'], axis=1)
df.head()

# iadelerin çıkarılması
df = df[~df["Invoice"].str.contains("C", na=False)]


df = df[(df["Quantity"] > 0)]
df = df[(df["Price"] > 0)]

# fatura basına ortalama kazanc
df["TotalPrice"] = df["Quantity"] * df["Price"]



###############################################################
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

# Recency (yenilik): Müşterinin son satın almasından bugüne kadar geçen süre
# Frequency (Sıklık): Toplam satın alma sayısı.
# Monetary (Parasal Değer): Müşterinin yaptığı toplam harcama.

df["InvoiceDate"].max()
today_date = dt.datetime(2010, 12, 11)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: num.nunique(),
                                     "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ['recency', 'frequency', 'monetary']


rfm.head()
rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]



##########
# K-Means
##########

sc = MinMaxScaler((0,1))

rfm_ = pd.DataFrame(sc.fit_transform(rfm), index=rfm.index, columns=rfm.columns)

distortions = []

K = range(1,10)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(rfm_)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Daha otomatik bir yol:
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(rfm_)
elbow.show()

elbow.elbow_value_

kmean = KMeans(n_clusters=6)
kmean.fit(rfm_)
labels = kmean.labels_
rfm_['Labels'] = labels

rfm_.groupby('Labels').mean()
