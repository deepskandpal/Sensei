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

def train(request):
    df = pd.read_csv('data.csv')
    isolation_forest = IsolationForest(n_estimators=100)
    isolation_forest.fit(df['Sales'].values.reshape(-1, 1))
    xx = np.linspace(df['Sales'].min(), df['Sales'].max(), len(df)).reshape(-1, 1)
    anomaly_score = isolation_forest.decision_function(xx)
    print(anomaly_score)
    outlier = isolation_forest.predict(xx)
    plt.figure(figsize=(10, 4))
    plt.plot(xx, anomaly_score, label='anomaly score')
    plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                     where=outlier == -1, color='r',
                     alpha=.4, label='outlier region')
    plt.legend()
    plt.ylabel('anomaly score')
    plt.xlabel('Sales')
    df.loc[df['Sales'] > 1500].head()
    fig = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return render(request, 'train.html', {'data': uri})


def index(request):
    return HttpResponse("Hello, world. You're at the modeltrain index.")