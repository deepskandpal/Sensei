from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse,HttpResponseRedirect
from django.contrib import messages
from django.shortcuts import redirect
from django.urls import reverse
import logging
import pandas as pd
import numpy as np
from numpy import percentile
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import io
import urllib, base64


def handle_uploaded_file(f):
    with open('data.csv', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def upload_csv(request):
    data = {}
    if "GET" == request.method:
        return render(request, "modeltrain/upload_csv.html", data)
    # if not GET, then proceed
    try:
        csv_file = request.FILES["csv_file"]
        if not csv_file.name.endswith('.csv'):
            messages.error(request,'File is not CSV type')
            return HttpResponseRedirect(reverse("modeltrain:upload_csv"))
        # if file is too large, return
        if csv_file.multiple_chunks():
            messages.error(request,"Uploaded file is too big (%.2f MB)." % (csv_file.size/(1000*1000),))
            return HttpResponseRedirect(reverse("modeltrain:upload_csv"))

        try:
            handle_uploaded_file(csv_file)
            return redirect("modeltrain:train")
        except Exception as e:
            logging.getLogger("error_logger").error(repr(e))
            pass

    except Exception as e:
        logging.getLogger("error_logger").error("Unable to upload file. "+repr(e))
        messages.error(request,"Unable to upload file. "+repr(e))

    return HttpResponseRedirect(reverse("modeltrain:upload_csv"))


def plot_graph(anomaly_score, xx, outlier, variable):
    plt.figure(figsize=(10, 4))
    plt.plot(xx, anomaly_score, label='anomaly score')
    plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                     where=outlier == -1, color='r',
                     alpha=.4, label='outlier region')
    plt.legend()
    plt.ylabel('anomaly score')
    plt.xlabel(variable)
    fig = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri


def plot_contour(df):
    minmax = MinMaxScaler(feature_range=(0, 1))
    df[['Sales', 'Profit']] = minmax.fit_transform(df[['Sales', 'Profit']])
    df[['Sales', 'Profit']].head()
    X1 = df['Sales'].values.reshape(-1, 1)
    X2 = df['Profit'].values.reshape(-1, 1)
    X = np.concatenate((X1, X2), axis=1)
    outliers_fraction = 0.01
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    clf = IForest(contamination=outliers_fraction, random_state=0)
    clf.fit(X)
    # predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1

    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(8, 8))
    # copy of dataframe
    df1 = df
    df1['outlier'] = y_pred.tolist()

    # sales - inlier feature 1,  profit - inlier feature 2
    inliers_sales = np.array(df1['Sales'][df1['outlier'] == 0]).reshape(-1, 1)
    inliers_profit = np.array(df1['Profit'][df1['outlier'] == 0]).reshape(-1, 1)

    # sales - outlier feature 1, profit - outlier feature 2
    outliers_sales = df1['Sales'][df1['outlier'] == 1].values.reshape(-1, 1)
    outliers_profit = df1['Profit'][df1['outlier'] == 1].values.reshape(-1, 1)

    print('OUTLIERS: ', n_outliers, 'INLIERS: ', n_inliers)

    # threshold value to consider a datapoint inlier or outlier
    threshold = percentile(scores_pred, 100 * outliers_fraction)

    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
    # fill blue map colormap from minimum anomaly score to threshold value
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)

    # draw red contour line where anomaly score is equal to thresold
    a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
    b = plt.scatter(inliers_sales, inliers_profit, c='white', s=20, edgecolor='k')

    c = plt.scatter(outliers_sales, outliers_profit, c='black', s=20, edgecolor='k')

    plt.axis('tight')
    plt.legend([a.collections[0], b, c], ['learned decision function', 'inliers', 'outliers'],
               prop=matplotlib.font_manager.FontProperties(size=20), loc='lower right')

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title('Isolation Forest')
    fig = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri, n_outliers, n_inliers


def train(request):
    df = pd.read_csv('data.csv')
    isolation_forest = IsolationForest(n_estimators=100)
    isolation_forest.fit(df['Sales'].values.reshape(-1, 1))
    xx = np.linspace(df['Sales'].min(), df['Sales'].max(), len(df)).reshape(-1, 1)
    anomaly_scores = isolation_forest.decision_function(xx)
    outlier_detected = isolation_forest.predict(xx)
    isolation_forest = IsolationForest(n_estimators=100)
    isolation_forest.fit(df['Profit'].values.reshape(-1, 1))
    xx = np.linspace(df['Profit'].min(), df['Profit'].max(), len(df)).reshape(-1, 1)
    anomaly_score = isolation_forest.decision_function(xx)
    outlier = isolation_forest.predict(xx)
    uri2 = plot_graph(anomaly_score, xx, outlier, 'Profit')
    uri3, n_outliers, n_inliers = plot_contour(df)
    outlier = df.loc[df['outlier'] == 1]
    outlier['City'].value_counts().plot(kind='bar')
    plt.title('Cities')
    plt.legend([''],
               prop=matplotlib.font_manager.FontProperties(size=20))
    plt.xlim((0, 20))
    plt.ylim((0, 20))
    fig = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri4 = urllib.parse.quote(string)
    uri5 = plot_bar_graph(outlier)
    uri = plot_bar_graph_2(outlier)
    uri2 = plot_graph(anomaly_scores, xx, outlier_detected, 'Sales')
    header = 'OUTLIERS: ' + str(n_outliers) + 'INLIERS: '+ str(n_inliers)
    return render(request, 'train.html', {'data': uri,
                                          'old_data': uri2,
                                          'new_data': uri5,
                                          'fast_data': uri3,
                                          'exact_data': uri4
                                          })


def plot_bar_graph(outlier):
    outlier.groupby('Sub-Category').size().plot(kind='bar')
    plt.title('Sub-Categories')
    plt.xlim((1, 9))
    plt.ylim((0, 30))
    fig2 = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    string2 = base64.b64encode(buf2.read())
    uri5 = urllib.parse.quote(string2)
    return uri5


def plot_bar_graph_2(outlier):
    outlier['Ship Mode'].value_counts().plot(kind='bar')
    plt.title('Shipings Mode')
    plt.xlim((0, 3))
    plt.ylim((0, 60))
    fig2 = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    string2 = base64.b64encode(buf2.read())
    uri5 = urllib.parse.quote(string2)
    return uri5


def index(request):
    return HttpResponse("Hello, world. You're at the modeltrain index.")