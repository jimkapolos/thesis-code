import os
import pandas as pd
from numpy import array
import scipy.stats as stats
import natsort
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


x=1
for i in range(7,8) :
     os.chdir(r"C:\Users\JIM\Desktop\thesis code\rnd\2013-"+str(i))
     if x!= 0:
          j=str(x)+"new.csv"
          file = pd.read_csv(j)

          time = file['Timestamp [ms]']
          cpu_usage = file["CPU usage [%]"]
          cpu_capacity = file['CPU capacity provisioned [MHZ]']
          cpu_cores = file['CPU cores']

          fig= px.scatter(file, x=time, y=cpu_usage)
          fig.show()

          fig1 = px.scatter(file, x=time, y=cpu_usage)
          fig1.update_layout(xaxis_range=[1372714712,1372720112])
          fig1.update_layout(title_text="ΑΠΟ ΤΙΜΗ 285-303 του CSV")
          fig1.show()


          #check time and values
          a=[]
          for i in range(0,8259):

               timediff=time[i+1]-time[i]

               a.append(timediff)
               #print(timediff)
               if timediff==110414:print("θεση στο txt(διαφορα 110414)",i+2,"\n")
               if timediff==3300:print("θεση στο txt(διαφορα 3300)",i+2,"\n")
          v = pd.DataFrame(a)
          print("v", v)
          print("v describe", v.describe())
          # s=v.sort_values()
          print("s", v.sort_values(by=0))




