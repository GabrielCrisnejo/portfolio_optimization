#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import scipy.optimize as op

def load_timeseries(ticker, directory='~/portfolio_optimization/data/'):
    
    path = os.path.expanduser(os.path.join(directory, f'{ticker}.csv'))
    raw_data = pd.read_csv(path)
    
    # Especificamos el formato de la fecha
    # Esto puede depender de donde bajamos las series de tiempo.
    # Acá las baje de yahho finances
    
    date_format = '%Y-%m-%d'
    
    # Armamos un DataFrame con la fecha y el precio de cierre del activo
    t = pd.DataFrame({
        'date': pd.to_datetime(raw_data['Date'], format=date_format, dayfirst=True),
        'close': raw_data['Close']})
    
    # Agregamos una columna con el retorno diario
    t = t.sort_values(by='date').reset_index(drop=True)
    t['return'] = t['close'].pct_change().dropna()
    
    return t.dropna()


# Cómo lo aplicamos a un activo en particular
# AAPL = load_timeseries('AAPL')

# Veamos como generarnos un DataFrame que tenga la siguiente estructura:
# Columna 0: date
# Columna i: return para el i-ésimo activo (i = 1, 2, ..., i_max=len(tickers))

def df_tickers(tickers):
    
    # Cargamos las series temporales de cada activo en un diccionario
    dic_timeseries = {ticker: load_timeseries(ticker) for ticker in tickers}
    
    # Hacemos la intersección de todas las fechas disponibles.
    # De esta forma nos aseguramos que vamos a considerar los retornos
    # diarios de cada activo para el mismo día, y por lo tanto podemos llegar a 
    # tener menos datos
    timestamps = set.intersection(*[set(t['date']) for t in dic_timeseries.values()])
    
    # Creamos un DataFrame con las fechas comunes
    df = pd.DataFrame({'date': sorted(timestamps)})
    
    # Agregamos la columna de retornos para cada ticker
    for ticker, ts in dic_timeseries.items():
        ts_filtered = ts[ts['date'].isin(timestamps)].sort_values(by='date').reset_index(drop=True)
        df[ticker] = ts_filtered['return'].values
    
    return df

# Veamos este nuevo DataFrame en nuestro conjunto de activos:
# tickers = AAPL, GS, NFLX, NVDA, JPM

tickers = ['AAPL', 'AMZN', 'NFLX', 'GOOGL', 'META', 'MSFT']
df_tick = df_tickers(tickers)

# Calculemos la matriz de covarianza y el retorno promedio anual de cada activo
cov = np.cov(df_tick.drop(columns=['date']), rowvar=False) *252.0
mean_return = df_tick.drop(columns=['date']).mean().values *252.0
var_return = df_tick.drop(columns=['date']).var().values *252.0

# --------------- Optimizacion -------------------

# Target anual return
target_return = 0.3  

# Funciones objetivo
def portfolio_variance(x, cov):
    return x.T @ cov @ x

def portfolio_return(x, mean_return):
    return -x.T @ mean_return


# Inicialización y restricciones comunes
x0 = np.full(len(tickers), 1 / np.sqrt(len(tickers)))
l1_norm = [{"type": "eq", "fun": lambda x: np.sum(np.abs(x)) - 1}]
positive_weights = [(0, None)] * len(tickers)

# Optimización de varianza mínima
def optimize_portfolio_min(cov, mean_return, target_return):
    returns_constraint = [{"type": "eq", "fun": lambda x: x.T @ mean_return - target_return}]
    constraints = l1_norm + returns_constraint

    result = op.minimize(portfolio_variance, x0, args=(cov,), constraints=constraints, bounds=positive_weights)
    return result.x, result.fun

# Optimización de retorno máximo
def optimize_portfolio_max(cov, mean_return, target_dispersion):
    dispersion_constraint = [{"type": "eq", "fun": lambda x: x.T @ cov @ x - target_dispersion}]
    constraints = l1_norm + dispersion_constraint

    result = op.minimize(portfolio_return, x0, args=(mean_return,), constraints=constraints, bounds=positive_weights)
    return result.x, -result.fun

# Ejecutamos la optimización de varianza mínima
optimize_vector_min, disp_obtained = optimize_portfolio_min(cov, mean_return, target_return)

# Crear DataFrame para varianza mínima
df_weights_min = pd.DataFrame({
    'tickers': tickers,
    'optimize_vector': optimize_vector_min
})

# Ejecutamos la optimización de retorno máximo
target_dispersion = disp_obtained
optimize_vector_max, return_obtained = optimize_portfolio_max(cov, mean_return, target_dispersion)

# Crear DataFrame para retorno máximo
df_weights_max = pd.DataFrame({
    'tickers': tickers,
    'optimize_vector': optimize_vector_max
})









