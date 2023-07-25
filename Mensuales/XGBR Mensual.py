
import xgboost as xgb
import pandas as pd

import tools
import graph

# Importar la Data
dat = pd.read_excel("Data Mensual Post-Análisis.xlsx",sheet_name="Data")
dat = dat.drop(["Año","Mes"], axis = 1)
dat = dat.rename(columns = {"Precio mayorista pollo vivo - Lima": 'Precio'}, inplace = False)

# Eliminar las Variables Anuales
anuales = dat.loc[:,["Consumo per capita de pollo","Producción nacional de jurel ","Producción nacional de bonito"]]
dat = dat.drop(["Consumo per capita de pollo","Producción nacional de jurel ","Producción nacional de bonito"],axis = 1)

####Creación de columna alimentos (Conversión Redondos)
dat["Alimentos"] = round(dat["Maíz EEUU"]*0.63+dat["Harina (torta) Soya EEUU"]*0.26+dat["Frejol de Soya EEUU"]*0.11,1)



varIn = ['Fecha',
 'Precio',
 'Carga Pollo BB Peru',
#'Inversión privada',
'Expectativas de inflación a mismo año',
 'Precio promedio al consumidor de huevos a granel (gallina)',
 'Inflación promedio en Lima',
 'Alimentos'
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
	"Modelo" : "XGB",
	"base_score": 0.5, 
	"booster": 'gbtree', 
	"colsample_bylevel": 1,
	"colsample_bynode": 1, 
	"colsample_bytree": 1, 
	"gamma": 0,
	"importance_type": 'gain', 
	"learning_rate": 0.1, 
	"max_delta_step": 0,
	"max_depth": 3, 
	"min_child_weight": 1, 
	"n_estimators": 100,
	"n_jobs": 1, 
	"nthread": None, 
	"objective": 'reg:linear', 
	"random_state": 0,
	"reg_alpha": 0, 
	"reg_lambda": 1, 
	"scale_pos_weight": 1, 
	"seed": None,
	"silent": None, 
	"subsample": 1, 
	"verbosity": 1
}



XGBR = xgb.XGBRegressor(base_score= modelo["base_score"], booster= modelo["booster"], colsample_bylevel= modelo["colsample_bylevel"],
       colsample_bynode= modelo["colsample_bynode"], colsample_bytree= modelo["colsample_bytree"], gamma= modelo["gamma"],
       importance_type= modelo["importance_type"], learning_rate= modelo["learning_rate"], max_delta_step= modelo["max_delta_step"],
       max_depth= modelo["max_depth"], min_child_weight= modelo["min_child_weight"], n_estimators= modelo["n_estimators"],
       n_jobs= modelo["n_jobs"], nthread= modelo["nthread"], objective= modelo["objective"], random_state= modelo["random_state"],
       reg_alpha= modelo["reg_alpha"], reg_lambda= modelo["reg_lambda"], scale_pos_weight= modelo["scale_pos_weight"], seed= modelo["seed"],
       silent=modelo["silent"], subsample= modelo["subsample"], verbosity= modelo["verbosity"])


#########Entrenamiento
XGBR.fit(x_train,y_train)

# tools.make_predictor_for_model(XGBR, modelo, varIn, "")


########Predicción

y_predTrain = XGBR.predict(x_train)

y_predTest = XGBR.predict(x_test)

tools.save_object("model_x.pkl", XGBR)
model_x=tools.load_object("model_x.pkl")
y_predTest_2 = model_x.predict(x_test)

y_realTrain = pd.DataFrame(y_train)

y_realTest = pd.DataFrame(y_test)

###################################

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