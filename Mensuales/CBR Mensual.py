
import pandas as pd
import catboost as cbr

import tools
import graph

# Importar la Data
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
  'Carga Pollo BB Peru',
  'Inversión privada',
  'Expectativas de inflación a mismo año',
  'Precio promedio al consumidor de huevos a granel (gallina)',
  'Inflación promedio en Lima',
  'Alimentos',
]

#################
#Preparando Data#
#################

df = dat.loc[:,varIn]
#df["Precio"] = df["Precio"].rolling(4).mean()

horizont = 1
df = tools.lag(df, horizont+0)
varIn.remove("Precio")
varIn.remove("Fecha")
df = df.drop(varIn,axis = 1)
df = df.dropna()

col_list=[]
for col_name in df.columns:
    if col_name == "Fecha" or col_name == "Precio" or  (col_name[-1].isnumeric() and int(col_name[-1]) >= horizont):
        col_list.append(col_name)
df = df[col_list]

###División de la Data
train_split= 0.80
target = "Precio"
time_col = "Fecha"

x_train, y_train, x_test, y_test, trainFecha, testFecha = tools.split_data(df, train_split, target, time_col)


########
#Modelo#
########

modelo ={
    "Modelo" : "CBR",
    "iterations" : None,
    "learning_rate" : None,
    "depth" : None,
    "l2_leaf_reg" : None,
    "model_size_reg" : None,
    "rsm" : None,
    "loss_function" : "RMSE",
    }


CBR = cbr.CatBoostRegressor(iterations=modelo["iterations"],
                        learning_rate=modelo["learning_rate"],
                        depth=modelo["depth"],
                        l2_leaf_reg=modelo["l2_leaf_reg"],
                        model_size_reg=modelo["model_size_reg"],
                        rsm=modelo["rsm"],
                        loss_function=modelo["loss_function"])

#########Entrenamiento
CBR.fit(x_train,y_train)



#tools.make_predictor_for_model(CBR, modelo, varIn,"_mm_1")



################################### Predicción
y_predTrain = CBR.predict(x_train)

y_predTest = CBR.predict(x_test)

y_realTrain = pd.DataFrame(y_train)

y_realTest = pd.DataFrame(y_test)


results_df = tools.make_results_df(y_predTrain, y_predTest, y_realTrain, 
                                   y_realTest, trainFecha, testFecha, modelo)


train_metrics = tools.metrics(y_predTrain, y_train)
real_metrics = tools.metrics(y_predTest, y_test)
print("\n train_metrics")
print(train_metrics)
print("\n real_metrics")
print(real_metrics)


print(modelo)
tools.save_model(modelo, results_df, train_metrics, real_metrics, varIn)


graph.plot_series_df(results_df,
                      ["Real", 
                      modelo["Modelo"]+"_test",
                      modelo["Modelo"]+"_train"], 
                      title= "Predicción",
                      save = True,
                      modelo = modelo["Modelo"])

graph.plot_series_df(results_df.iloc[-len(y_test):],
                      ["Real",
                      modelo["Modelo"]+"_test",
                      ],
                      title= "Predicción Detalle",
                      save = True,
                      modelo = modelo["Modelo"])

graph.plot_scatter_pred(results_df.iloc[-len(y_test):],
                      "Real", 
                      modelo["Modelo"]+"_test",
                      title= "Dispersión de Predicción",
                      save = True,
                      modelo = modelo["Modelo"])

