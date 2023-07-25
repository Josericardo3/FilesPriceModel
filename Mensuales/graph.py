import matplotlib.pyplot as plt
# import plotly.figure_factory as ff
import seaborn as sns
import pandas as pd
import numpy as np

#plt.style.use('dark_background')
plt.style.use('default')


def plot_series_df(fc, cols_list, title="", save=False, modelo=""):
	plt.figure(figsize=(12, 6), dpi=300)
	for dtype in cols_list:
	    plt.plot(
	    fc.index,
	    dtype,
	    '.-',
	    data=fc,
	    label=dtype,
	    alpha=1,
	    linewidth=2.0,
	    )
	title = title+" "+modelo
	plt.title(title)
	plt.xlabel('Linea de Tiempo')
	plt.ylabel('Precio del Pollo')
	plt.grid()
	if save:
			plt.savefig("./Models_Results_m/"+modelo+'/'+title+'.png')

def plot_scatter_pred(fc, real, pred, title="", save=False, modelo=""):
	title = title+" "+modelo
	plt.figure(figsize=(10, 5))
	plt.scatter(fc[real], fc[pred])
	plt.plot(fc[real], fc[real], 'k--')
	plt.title(title)
	plt.xlabel('Real')
	plt.ylabel('Pronóstico')
	#plt.legend()
	plt.grid()
	if save:
			plt.savefig("./Models_Results_m/"+modelo+'/'+title+'.png')
	



def plot_residuos_analisis(real, predict):
	y_train = real
	prediccion_train = predict
	residuos_train = real - predict
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 11))
	axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
	axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
	                'k--', color = 'black', lw=2)
	axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
	axes[0, 0].set_xlabel('Real')
	axes[0, 0].set_ylabel('Predicción')
	axes[0, 0].tick_params(labelsize = 7)

	axes[0, 1].bar(list(range(len(y_train))), residuos_train, alpha = 0.4, color = 'blue')
	axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
	axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
	axes[0, 1].set_xlabel('Tiempo')
	axes[0, 1].set_ylabel('Residuo')
	axes[0, 1].tick_params(labelsize = 7)
	
	sns.distplot(residuos_train, ax=axes[1, 0])
	axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10, fontweight = "bold")
	axes[1, 0].set_xlabel("Residuo")
	axes[1, 0].tick_params(labelsize = 7)

	axes[1, 1].scatter(prediccion_train, residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
	axes[1, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
	axes[1, 1].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
	axes[1, 1].set_xlabel('Predicción')
	axes[1, 1].set_ylabel('Residuo')
	axes[1, 1].tick_params(labelsize = 7)

	fig.tight_layout()
	plt.subplots_adjust(top=0.9)
	fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");

