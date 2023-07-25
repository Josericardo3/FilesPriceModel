
import pandas as pd
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

import graph
import tools


dat = pd.read_excel("Variables Diarias Consolidadas.xlsx",sheet_name="Sheet1")
dat = dat.drop(["Precio Max","Precio Min"], axis = 1)


#################
#Preparando Data#
#################

df = dat.copy()
#df["Precio"] = df["Precio"].rolling(4).mean()

horizont = 15

df = tools.p30(df,horizont)

df["f_Fecha"] = df["Fecha"].shift(-horizont)
df["Fecha"] = df["f_Fecha"]
df = df.drop("f_Fecha", axis=1)

df1 = df.copy()
y_max = df1["Precio+"+str(horizont)].max()
y_min = df1["Precio+"+str(horizont)].min()
df1,xmin,xmax = tools.norm(df1)
df = df1.copy()

# l = 1
# df = tools.lag(df,l)

df = df.interpolate(method="nearest")
df = df.dropna()

################################################ALIMENTOS

df["Alimentos"] = df["Futuros Maiz (Ajd. Close)"]*0.63+df["Futuros Harina (Adj. Close)"]*0.26+df["Futuros Frijol (Adj. Close)"]*0.11
df = df.drop(["Futuros Maiz (Ajd. Close)", "Futuros Harina (Adj. Close)", "Futuros Frijol (Adj. Close)","Peso Promedio"], axis=1)

df = df.reset_index(drop=True)



##############################################################################

rezago = 30 # numero de rezagos

varIn = df.columns
varIn = (df.columns).drop(["Fecha","Precio+"+str(horizont)])

xTotal , yTotal, fechasTotales = tools.lagLSTM(df,rezago,obj = "Precio+"+str(horizont))

train_split= 0.80
x_train, y_train, x_test, y_test, trainFecha, testFecha = tools.split_data_LSTM(xTotal, 
                                                                          yTotal, fechasTotales, train_split, df)

############
#Modelo
############
modelo = {
	"Modelo" : "LSTM",
	"loss" : 'mse',
	"optimizer" : 'adam',
	"metrics" : ['mae'],
	"epochs" : 100,
	"batch_size" : 32,
	"validation_split" : 0.2
	}


n_steps = rezago + 1
n_features = xTotal.shape[2]

model = Sequential()
model.add(LSTM(10, return_sequences=False, input_shape=(n_steps, n_features)))
#model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer=modelo['optimizer'],
              loss=modelo['loss'],metrics=modelo['metrics'])

#########Entrenamiento

es = EarlyStopping(monitor='val_loss',patience=100,mode='min',
                   restore_best_weights=True,min_delta=500,verbose=0)

history = model.fit(x_train,y_train, batch_size=32,
epochs=100, validation_split = 0.2,callbacks=[es],verbose=1,shuffle=False)


#########Predicción
y_predTrain = model.predict(x_train)

y_predTest = model.predict(x_test)

y_predTrain= y_predTrain.reshape(len(y_predTrain),)
y_predTest = y_predTest.reshape(len(y_predTest),)

#desnormalizando
y_predTrain = (y_predTrain*(y_max - y_min)) + y_min
y_train = (y_train*(y_max - y_min)) + y_min
y_predTest = (y_predTest*(y_max - y_min)) + y_min
y_test = (y_test*(y_max - y_min)) + y_min

y_realTrain = pd.DataFrame(y_train)

y_realTest = pd.DataFrame(y_test)



results_df = tools.make_results_df_LSTM(y_predTrain, y_predTest, y_realTrain, 
                                   y_realTest, trainFecha, testFecha, modelo)

train_metrics = tools.metrics(y_predTrain, y_train.reshape(len(y_train)))
real_metrics = tools.metrics(y_predTest, y_test.reshape(len(y_test)))
print("\n train_metrics")
print(train_metrics)
print("\n real_metrics")
print(real_metrics)


print(modelo)

tools.save_model(modelo, results_df, train_metrics, real_metrics,varIn)

graph.plot_series_df(results_df,
                      ["Real", 
                      modelo["Modelo"]+"_test",
                      modelo["Modelo"]+"_train"], 
                      title= "Predicción",
                      save = True,
                      modelo = modelo["Modelo"])

graph.plot_series_df(results_df.iloc[-len(y_test):],
                      ["Real", 
                      modelo["Modelo"]+"_test"],
                      title= "Predicción Detalle",
                      save = True,
                      modelo = modelo["Modelo"])

graph.plot_scatter_pred(results_df.iloc[-len(y_test):],
                      "Real", 
                      modelo["Modelo"]+"_test",
                      title= "Dispersión de Predicción",
                      save = True,
                      modelo = modelo["Modelo"])

######## LO SIGUIENTE ES POR SI SE QUIERE HACER UN RESAMPLE (CAMBIAR LAS PREDICCIONES DIARIAS A SEMANALES O MENSUALES)

# results_df = results_df.iloc[-len(y_test):]
# # weekly = results_df.resample('W').mean()
# monthly = results_df.resample('M').mean()

# train_metrics = tools.metrics(y_predTrain, y_train)
# real_metrics = tools.metrics(y_predTest, y_test)

# name_test = ""
# for i in results_df.columns:
#     if i.endswith('_test'):
#         name_test=i
        
# resamp_metrics = tools.metrics(monthly["Real"], monthly[name_test])

# print("\n train_metrics")
# print(train_metrics)
# print("\n real_metrics")
# print(real_metrics)
# print("\n resamp_metrics")
# print(resamp_metrics)


# print(modelo)
# tools.save_model(modelo, results_df, train_metrics, real_metrics, resamp_metrics)

# graph.plot_series_df(monthly,
#                       ["Real", 
#                       modelo["Modelo"]+"_test"],
#                       title= "Predicción Detalle R_M",
#                       save = True,
#                       modelo = modelo["Modelo"])

# graph.plot_scatter_pred(monthly,
#                       "Real", 
#                       modelo["Modelo"]+"_test",
#                       title= "Dispersión de Predicción R_M",
#                       save = True,
#                       modelo = modelo["Modelo"])

