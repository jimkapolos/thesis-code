import os
import pandas as pd
import scipy.stats as stats
import natsort
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# file 1.csv and 2.csv
x=1
for i in range(7,8) :
     os.chdir(r"C:\Users\JIM\Desktop\thesis code\rnd\2013-"+str(i))
     if x!= 0:
          j=str(x)+".csv"
          file = pd.read_csv(j)

          time = file['Timestamp [ms]']
          cpu_usage = file["CPU usage [%]"]
          cpu_capacity = file['CPU capacity provisioned [MHZ]']
          cpu_cores = file['CPU cores']

#          MINMAXSCALER

          scaler = MinMaxScaler()
          data_scaled = scaler.fit_transform(file.to_numpy())
          data_scaled = pd.DataFrame(data_scaled,
                                     columns=['Timestamp [ms]', 'CPU cores', 'CPU capacity provisioned [MHZ]','CPU usage [MHZ]',
                                              'CPU usage [%]', 'Memory capacity provisioned [KB]', 'Memory usage [KB]',
                                              'Disk read throughput [KB/s]', 'Disk write throughput [KB/s]',
                                              'Network received throughput [KB/s]',
                                              'Network transmitted throughput [KB/s]'
                                              ])
          time1 = data_scaled['Timestamp [ms]']
          cpu_usage1 = data_scaled["CPU usage [%]"]
          cpu_capacity1 = data_scaled['CPU capacity provisioned [MHZ]']
          cpu_cores1 = data_scaled['CPU cores']
          # time = file['Timestamp [ms]'] *0.001 #second
          # time = file['Timestamp [ms]'] *0.000016667 #minutes



          fig1 = px.scatter(data_scaled, x=time1, y=cpu_usage1, color=cpu_capacity1)

          fig1.update_layout(title_text='MINMAXSCALING FILE 2013-7:' + str(x) + '.csv')

          fig1.show()

          # ------------------------------------------------------------------------------------------
          fig = make_subplots(rows=1, cols=2)

          fig.add_trace(go.Scatter(x=time, y=cpu_usage, name='CPU usage [%]', mode='markers'),
                        row=1, col=1
                        )
          fig.add_trace(go.Scatter(x=time1, y=cpu_usage1, name='MINMAXSCALED CPU usage [%]', mode='markers'),
                        row=1, col=2)
          fig.update_xaxes(title_text="Timestamp [ms]")
          fig.update_yaxes(title_text="CPU usage [%]")
          fig.update_layout(title_text='FILE 2013-7:' + str(x) + '.csv')
          fig.show()
          # ---------------------------------------------------------------------------------

          fig2 = make_subplots(rows=1, cols=2)

          fig2.add_trace(go.Scatter(x=time, y=cpu_capacity, name='CPU capacity provisioned [MHZ]', mode='markers'),
                         row=1, col=1
                         )
          fig2.add_trace(
               go.Scatter(x=time1, y=cpu_capacity1, name='MINMAXSCALED CPU capacity provisioned [MHZ]', mode='markers'),
               row=1, col=2)
          fig2.update_xaxes(title_text="Timestamp [ms]")
          fig2.update_yaxes(title_text="CPU capacity provisioned [MHZ]")

          fig2.update_layout(title_text='FILE 2013-7:' + str(x) + '.csv')
          fig2.show()


#              AVERAGE
          print(cpu_usage.describe(),"\n")
          print(cpu_capacity.describe(),'\n')
          print(cpu_cores.describe(),'\n')
          print(time.describe(),'\n')

          sum1 = 0
          sum2 = 0
          sum3 = 0
          sum4 = 0
          N=len(file)
          print('στοιχεια του αρχειου:',N,'\n')
          for i in range(N):
               sum1 = sum1 + file['CPU usage [%]'][i]
               sum2 = sum2 + file['CPU usage [MHZ]'][i]
               sum3 = sum3 + file['CPU capacity provisioned [MHZ]'][i]
               sum4 = sum4 + file['Timestamp [ms]'][i]

          average1 = sum1 / N
          average1_1 = sum2 / N
          average2 = sum3 / N
          average3 = sum4 / N

          print('Αverage Timestamp [ms]:',average3, '\n')
          print('Αverage CPU usage [%]:',average1, '\n')
          print('Αverage CPU usage [MHZ]:',average1_1, '\n')
          print('Αverage CPU capacity provisioned [MHZ]:', average2, '\n')


#              Dispersion measures

          for j in range(N):
               sum1_1= pow((file['CPU usage [%]'][j]-average1),2)
               sum2_2=pow((file['CPU usage [MHZ]'][j]-average1_1),2)
               sum3_3=pow((file['CPU capacity provisioned [MHZ]'][j]-average2),2)
               sum4_4=pow((file['Timestamp [ms]'][j]-average3),2)

          dispersion1 = sum1_1 / (N-1)
          dispersion1_1 = sum2_2 / (N-1)
          dispersion2 = sum3_3 / (N-1)
          dispersion3 = sum4_4 / (N - 1)

          print('Dispersion measures Timestamp [ms]:', dispersion3, '\n')
          print('Dispersion measures CPU usage [%]:', dispersion1, '\n')
          print('Dispersion measures CPU usage [MHZ]:', dispersion1_1, '\n')
          print('Dispersion measures CPU capacity provisioned [MHZ]:', dispersion2,'\n\n')

#         AFTER SCALING     AVERAGE

          sum1_scale = 0
          sum2_scale = 0
          sum3_scale = 0
          sum4_scale = 0


          for i in range(N):
               sum1_scale = sum1_scale + data_scaled['CPU usage [%]'][i]
               sum2_scale = sum2_scale + data_scaled['CPU usage [MHZ]'][i]
               sum3_scale = sum3_scale + data_scaled['CPU capacity provisioned [MHZ]'][i]
               sum4_scale = sum4_scale + data_scaled['Timestamp [ms]'][i]

          average1_scale = sum1_scale / N
          average1_1_scale = sum2_scale / N
          average2_scale = sum3_scale / N
          average3_scale = sum4_scale / N

          print("AFTER SCALING \n")
          print('Αverage Timestamp [ms]:',average3_scale, '\n')
          print('Αverage CPU usage [%]:',average1_scale, '\n')
          print('Αverage CPU usage [MHZ]:',average1_1_scale, '\n')
          print('Αverage CPU capacity provisioned [MHZ]:', average2_scale, '\n')



#            AFTER SCALING  Dispersion measures

          for j in range(N):
               sum1_1_scale= pow((data_scaled['CPU usage [%]'][j]-average1_scale),2)
               sum2_2_scale=pow((data_scaled['CPU usage [MHZ]'][j]-average1_1_scale),2)
               sum3_3_scale=pow((data_scaled['CPU capacity provisioned [MHZ]'][j]-average2_scale),2)
               sum4_4_scale=pow((data_scaled['Timestamp [ms]'][j]-average3_scale),2)

          dispersion1_scale = sum1_1_scale / (N-1)
          dispersion1_1_scale = sum2_2_scale / (N-1)
          dispersion2_scale = sum3_3_scale / (N-1)
          dispersion3_scale = sum4_4_scale / (N-1)

          print('Dispersion measures Timestamp [ms]:', dispersion3_scale, '\n')
          print('Dispersion measures CPU usage [%]:', dispersion1_scale, '\n')
          print('Dispersion measures CPU usage [MHZ]:', dispersion1_1_scale, '\n')
          print('Dispersion measures CPU capacity provisioned [MHZ]:', dispersion2_scale,'\n\n')

#         Check data-set for empty values

          print('Check data-set for empty values\n', file.isnull().sum(), '\n\n')
          print('\nCheck data-set for NaN values\n', file.isna().sum(), '\n\n')



#          Check data-set for outliers

          plot=make_subplots(rows=1, cols=2)
          plot.add_trace(go.Box(y=time,name='Timestamp [ms]'),row=1,col=1)
          plot.add_trace(go.Box(y=time1,name='Scale Timestamp [ms]'),row=1,col=2)
          plot.update_yaxes(title_text="Timestamp [ms]")
          plot.update_layout(title_text='FILE 2013-7:' + str(x) + '.csv' + '  Check outliers')
          plot.show()

          plot1 = make_subplots(rows=1, cols=2)
          plot1.add_trace(go.Box(y=cpu_usage,name='CPU usage [%]'), row=1, col=1)
          plot1.add_trace(go.Box(y=cpu_usage1,name='Scale CPU usage [%]'), row=1, col=2)
          plot1.update_yaxes(title_text="CPU usage [%]")
          plot1.update_layout(title_text='FILE 2013-7:' + str(x) + '.csv' + '  Check outliers')
          plot1.show()

          plot2 = make_subplots(rows=1, cols=2)
          plot2.add_trace(go.Box(y=cpu_capacity,name='CPU capacity provisioned [MHZ]'), row=1, col=1)
          plot2.add_trace(go.Box(y=cpu_capacity1,name='Scale CPU capacity provisioned [MHZ]'), row=1, col=2)
          plot2.update_yaxes(title_text="CPU capacity provisioned [MHZ]")
          plot2.update_layout(title_text='FILE 2013-7:' + str(x) + '.csv' + '  Check outliers')
          plot2.show()


