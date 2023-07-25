import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statsmodels.formula.api as smf
import os
import pickle
import shutil

from datetime import date
from datetime import datetime
from datetime import timedelta


def metrics(real, pred, cant_predictors=None):
	stats = pd.DataFrame()
	res = real - pred
	stats["SSD"] = [sum((res)**2)]
	#stats["RSE"] = [np.sqrt(stats["SSD"][0]/(len(real)-1-cant_predictors))]
	#stats["error"] = [stats["RSE"][0]/np.mean(real)]
	stats["R^2"] = [det_coef(real, pred)]
	stats["MSE"] = [mean_squared_error(real, pred)]
	stats["RMSE"] = [np.sqrt(mean_squared_error(real, pred))]
	stats["MAE"] = [mean_absolute_error(real, pred)]
	#stats["AIC"] = [len(real)*np.log(stats["SSD"]/len(real)) + (2*cant_predictors)]
	return stats

def fix_text_list(features):
	text = ""
	mult = 0
	for i in range(len(features)):
		if i < mult+2:
			text += features[i]+ "  |  "
		else:
			text += " \n "
			text += features[i]+ "  |  "
			mult = i
	return text


def det_coef(x,y):
    return r2_score(x,y)

def secuencial_train_test_split(data, split_p):
	split_index = round(len(data)*split_p)
	data_train = data.iloc[:split_index]
	data_test = data.iloc[split_index:]
	return data_train, data_test


def save_stats2excel(model, df_stats, name):
    writer = pd.ExcelWriter("model_stats/"+name+".xlsx", engine = 'xlsxwriter')
    summary = model.summary()
    html= summary.tables[0].as_html()
    summary_1 = pd.read_html(html)[0]
    summary_1.to_excel(writer, "stats1")
    html= summary.tables[1].as_html()
    summary_1 = pd.read_html(html)[0]
    summary_1.to_excel(writer, "stats2")
    model.pvalues.to_excel(writer, "stats3")
    df_stats.to_excel(writer, "stats4")
    writer.save()
    writer.close()

def normalize_2(data_u):
    data_max = data_u.max(axis=0)
    data_min = data_u.min(axis=0)
    data_dif = data_max - data_min
    data_u = (data_u - data_min)/data_dif
    return data_u, data_min, data_dif

def normalize(data_u):
	data_mean = data_u.mean(axis=0)
	data_std = data_u.std(axis=0)
	data_u = (data_u - data_mean)/data_std
	return data_u, data_mean, data_std


def desnormalize(data, mean, std):
    des_norm = (data*std)+mean
    return des_norm

def desnormalize_2(data, data_min, data_dif):
    des_norm = (data*data_dif) + data_min
    return des_norm

# html= summary.tables[0].as_html()
# summary = pd.read_html(html)[0]

def load_perf(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def save_perf(name, perf):
    f = open(name,"wb")
    pickle.dump(perf,f)
    f.close()


def load_object(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def save_object(name, obj):
    f = open(name,"wb")
    pickle.dump(obj,f)
    f.close()


def save_model(modelo, results_df, train_metrics, real_metrics, varIn=[]):
    dir_name = "./Models_Results_m/"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    dir_name = dir_name+modelo["Modelo"]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    model = pd.DataFrame(data=modelo, index=[0]).T
    writer = pd.ExcelWriter(dir_name+"/Metrics_Params_"+modelo["Modelo"]+".xlsx", engine = 'xlsxwriter')
    train_metrics.to_excel(writer, "train_metrics")
    real_metrics.to_excel(writer, "real_metrics")
    pd.DataFrame(varIn).to_excel(writer, "input_vars")
    model.to_excel(writer, "modelo")
    writer.save()
    save_perf(dir_name+"/result_df_"+modelo["Modelo"]+'.pkl', results_df)


def make_predictor_for_model(model_t, modelo, varIn, name=""):
    dir_name = "./Models_Predictors/"+modelo["Modelo"]+"_predictor"+name
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    save_object(dir_name+"/"+modelo["Modelo"]+"_model.pkl", model_t)
    varIn.insert(0,'Precio')
    save_object(dir_name+"/"+modelo["Modelo"]+"_inputvars.pkl", varIn)
    varIn.insert(0,"Tiempo")
    input_df = pd.DataFrame(columns = varIn)
    none_row = [None for a in varIn]
    input_df = input_df.append(pd.Series(none_row, index=varIn), ignore_index=True)
    input_df.reset_index()
    input_df.to_excel(dir_name+"/"+modelo["Modelo"]+"_input.xlsx", "inputs", index=False)
    shutil.copy2("make_predicction.py", dir_name)

def make_predictor_for_model_LSTM(model_t, modelo, varIn, xmin, xmax, n_steps, name=""):
    dir_name = "./Models_Predictors/"+modelo["Modelo"]+"_predictor"+name
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    model_t.save(dir_name+"/"+modelo["Modelo"]+"_model.h5")
    varIn.insert(0,'Precio')
    metadata = {"varIn" : varIn, "max" : xmax, "min" : xmin, "n_steps" : n_steps}
    save_object(dir_name+"/"+modelo["Modelo"]+"_metadata.pkl", metadata)
    varIn.insert(0,"Tiempo")
    input_df = pd.DataFrame(columns = varIn)
    none_row = [None for a in varIn]
    input_df = input_df.append(pd.Series(none_row, index=varIn), ignore_index=True)
    input_df.reset_index()
    input_df.to_excel(dir_name+"/"+modelo["Modelo"]+"_input.xlsx", "inputs", index=False)
    shutil.copy2("make_predicction_LSTM.py", dir_name)


def split_data(df, train_split, target, time_col):
    train = df.head(int(len(df)*train_split))
    test = df.tail(int(len(df)*round(1-train_split,2)))
    
    trainFecha = train.loc[:,time_col]
    testFecha = test.loc[:,time_col]
    
    x_train = train.drop(target, axis=1) # separa las variables independientes
    x_train = x_train.drop(time_col, axis=1)
    y_train = train.loc[:,target] # separa la clasificacion
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    
    x_test = test.drop(target, axis=1) # separa las variables independientes
    x_test = x_test.drop(time_col, axis=1)
    y_test = test.loc[:,target] # separa la clasificacion
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    x_train = x_train.values.tolist()
    x_train = pd.DataFrame(x_train)
    
    x_test = x_test.values.tolist()
    x_test = pd.DataFrame(x_test)
    
    return x_train, y_train, x_test, y_test, trainFecha, testFecha

def lag(df,rezago):
    col = list(df.columns) #guarda columna en lista
    auxcol = len(df.columns)
    df = df.values.tolist() #transforma tabla
    
    for t in range(rezago):
        for item in range(1,auxcol): #para cada variable
            col.append(col[item]+" - " + str(t+1)) # inventa nueva columna
    
    for t in range(1,rezago+1):
        for fila in range(len(df)): #para cada fila
            for columna in range(1,auxcol): #para cada variable
                if fila < t: #si fila es menor que rezago
                    df[fila].append(np.nan) #agrega vacios
                else:
                    df[fila].append(df[fila-(t)][columna]) # agrega rezago
                
    df = pd.DataFrame(df) #dejalo en dataframe
    df.columns = col #deja el nombre de las variables
    return df

def make_results_df(y_predTrain, y_predTest, y_realTrain, y_realTest, trainFecha, testFecha, modelo):
    y_df_train = y_realTrain.set_index(trainFecha)
    y_df_test = y_realTest.set_index(testFecha)
    y_df = pd.concat([y_df_train, y_df_test])
    y_df = y_df.rename(columns = {"Precio": 'Real'}, inplace = False)
    
    results_df_train = pd.DataFrame()
    results_df_train[modelo["Modelo"]+"_train"] = y_predTrain
    results_df_train = results_df_train.set_index(trainFecha)
    
    results_df = pd.DataFrame()
    results_df[modelo["Modelo"]+"_test"] = y_predTest
    results_df = results_df.set_index(testFecha)
    results_df = y_df.join(results_df)
    results_df = results_df.join(results_df_train)
    
    return results_df



#-----------------
#LSTM
#-----------------

def lagLSTM(df,rezago,obj): # le da la forma que necesita la LSTM para recibir el input, un cubo
    
    yObjetivo = df.loc[:,obj] # separa valor a predecir
    df = df.drop(obj, axis=1) # remueve el valor a predecir de las variables
    fechas = df.loc[:,"Fecha"] # separa la fecha
    fechas = fechas.reset_index(drop=True) # reset a los indices
    df = df.drop("Fecha", axis=1) # bota la fecha d elos datos
    df = df.values.tolist() # datos a lista
    yObjetivo = yObjetivo.values.tolist() # precio a 30 dias a lista

    x = [] # lista auxiliar para guardar datos
    y = [] # lista auxiliar para guardar precio+30
    fx = [] #lista auxiliar para guardar fechas
    for i in range(rezago,len(df)): #para cada fila
        auxx=[] # lista auxiliar que recoje la semana de las variables
        auxy=[] # lista auxiliar que recoje la semana de los dolar+30
        for t in range(rezago+1): # para todos los rezagos
            auxx.append(df[i-rezago+t]) #agrega el valor de las variables para esa semana
        auxy.append(yObjetivo[i]) # agrega el valor del precio+30 para esa semana
        x.append(auxx) # las agrega finalmente a la lista x
        y.append(auxy) # lo mismo para el precio+30 en la lista y
        fx.append(fechas.iloc[i]) #guarda la fecha en fx

    return np.array(x), np.array(y), fx # retorna el cubo de variables, la lista de precio+30 y las fechas


def p30(df,d): # d significa el horizonte de pronostico (si es 30, pronostico a 30 dias)
    col = list(df.columns) #guarda columna en lista
    df = df.values.tolist() #transforma tabla
    doldi = {} #crea diccionario
        
    for fila in range(len(df)): #para cada fila
        if str(df[fila][1]) == "nan": #si el valor del precio es nan
            continue #siguiente
        else:        # si no es nan
            doldi[df[fila][0]] = df[fila][1]
    fech = doldi.keys()     
    
    for fila in range(len(df)): #para cada fila
        encontro = True #booleano para ciclo
        sep = 0 #auxiliar que va aumentando la tolerancia a la distancia de 30 dias
        while encontro: #mientras no encuentre
            if (df[fila][0]+timedelta(days=d+sep)) in fech: #si tenemos el precio en 30+sep dias
                df[fila].append(doldi[df[fila][0]+timedelta(days=d+sep)]) #agrega dato a la fila
                encontro = False #sale del loop
            elif (df[fila][0]+timedelta(days=d-sep)) in fech:#si tenemos el precio en 30-sep dias
                df[fila].append(doldi[df[fila][0]+timedelta(days=d-sep)]) #agrega el dato a la fila
                encontro = False #sale del loop
            elif df[fila][0] >= max(fech)-timedelta(days=d):#Vemos si ya llegamos a una fecha donde no tenemos el valor a 30 dias
                df[fila].append(np.nan)#Llenamos con nan
                encontro = False
            else:
                sep += 1 #en caso contrario, aumenta tolerancia
    
    col.append("Precio+" + str(d)) #agrega nombre de columna
    df = pd.DataFrame(df) #transforma lista en dataframe
    df.columns = col #nombres de las columnas
    return df

def norm(dfr):
    minimos = []
    maximos = []    
    col = list(dfr.columns)
    col.remove("Fecha")
    for i in col: 
        p = (dfr[i].copy())
        maxi = max((p.dropna()).values)
        maximos.append(maxi)
        mini = min((p.dropna()).values)
        minimos.append(mini)
        p = (p-mini)/(maxi-mini)
        dfr[i] = pd.DataFrame(p)
    return dfr,minimos,maximos

def make_results_df_LSTM(y_predTrain, y_predTest, y_realTrain, y_realTest, trainFecha, testFecha, modelo):
    y_realTrain["Fecha"] = trainFecha
    y_df_train = y_realTrain.set_index("Fecha")
    y_realTest["Fecha"] = testFecha
    y_df_test = y_realTest.set_index("Fecha")
    y_df = pd.concat([y_df_train, y_df_test])
    y_df.columns = ["Real"]

    results_df_train = pd.DataFrame()
    results_df_train[modelo["Modelo"]+"_train"] = y_predTrain
    results_df_train["Fecha"] = trainFecha
    results_df_train = results_df_train.set_index("Fecha")

    results_df = pd.DataFrame()
    results_df[modelo["Modelo"]+"_test"] = y_predTest
    results_df["Fecha"] = testFecha
    results_df = results_df.set_index("Fecha")
    results_df = y_df.join(results_df)
    results_df = results_df.join(results_df_train)
    return results_df

def split_data_LSTM(xTotal , yTotal, fechasTotales, train_split, df):
    trainFecha = fechasTotales[:int(len(df)*train_split)]
    testFecha = fechasTotales[int(len(df)*train_split):]
    x_train = xTotal[:int(len(df)*train_split),:,:]
    x_test = xTotal[int(len(df)*train_split):,:,:]
    y_train = yTotal[:int(len(df)*train_split),:]
    y_test = yTotal[int(len(df)*train_split):,:]
    return x_train, y_train, x_test, y_test, trainFecha, testFecha