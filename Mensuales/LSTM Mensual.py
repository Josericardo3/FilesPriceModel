
import pandas as pd

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import tools
import graph




dat = pd.read_excel("Data Mensual Post-Análisis.xlsx",sheet_name="Data")
dat = dat.drop(["Año","Mes"], axis = 1)
dat = dat.rename(columns = {"Precio mayorista pollo vivo - Lima": 'Precio'}, inplace = False)

# Eliminar las Variables Anuales
anuales = dat.loc[:,["Consumo per capita de pollo","Producción nacional de jurel ","Producción nacional de bonito"]]
dat = dat.drop(["Consumo per capita de pollo","Producción nacional de jurel ","Producción nacional de bonito"],axis = 1)

####Creando columna columna alimentos (Conversión Redondos)
dat["Alimentos"] = round(dat["Maíz EEUU"]*0.63+dat["Harina (torta) Soya EEUU"]*0.26+dat["Frejol de Soya EEUU"]*0.11,1)


varIn = ['Fecha',
 'Precio',

 #'Maíz EEUU',
 #'Harina (torta) Soya EEUU',
 #'Frejol de Soya EEUU',
 #'TC Bancario',
 'Carga Pollo BB Peru',
 #'Volúmen de venta de pollo vivo - Lima',


 'Inversión privada',
 'Expectativas de inflación a mismo año',
 #'Precio promedio al consumidor de jurel (Lima)',

 #'Precio promedio al consumidor de jurel (Nacional)',
 #'Precio promedio al consumidor de arroz corriente',
 'Precio promedio al consumidor de huevos a granel (gallina)',
 'Inflación promedio en Lima',
  'Alimentos'
 #'Precio promedio al consumidor de yuca blanca'
]

#################
#Preparando Data#
#################

df = dat.loc[:,varIn]
#df["Precio"] = df["Precio"].rolling(4).mean()

#Normalizando
df1 = df.copy()
y_max = df1["Precio"].max()
y_min = df1["Precio"].min()
df1,xmin,xmax = tools.norm(df1)
df = df1.copy()

l = 1
df = tools.lag(df,l)
varIn.remove("Fecha") 

if l > 1: 
    varAux = []
    for k in range(1,l): 
        for variable in varIn: 
            varAux.append(variable+" - "+str(k))
    for variable in varAux: 
        varIn.append(variable)
       
varIn.remove("Precio")   
df = df.drop(varIn,axis = 1)
df = df.dropna()
df = df.reset_index(drop=True) # reset a los index

##############################################################################

rezago = 2 # numero de rezagos
xTotal , yTotal, fechasTotales = tools.lagLSTM(df,rezago,obj = "Precio")

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
	"batch_size" : 1,
	"validation_split" : 0.2
	}

n_steps = rezago + 1
n_features = xTotal.shape[2]

model = Sequential()
model.add(LSTM(20, return_sequences=False, input_shape=(n_steps, n_features)))
#model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer=modelo['optimizer'],
              loss=modelo['loss'],metrics=modelo['metrics'])

#########Entrenamiento
history = model.fit(x_train,y_train, batch_size=modelo['batch_size'],
epochs=modelo['epochs'], validation_split =modelo['validation_split'],verbose=1,shuffle=True)


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
tools.save_model(modelo, results_df, train_metrics, real_metrics, varIn)

#tools.make_predictor_for_model_LSTM(model, modelo, varIn, xmin, xmax, n_steps)

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