

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import argparse
from typing import Tuple, List, Dict, Any
import json


class TimeSeriesForecaster:
    """
    Основной класс для прогнозирования временных рядов
    Автоматически определяет лучшую модель и параметры
    """
    
    def __init__(self, data: pd.Series, horizon: int):
        """
        Инициализация прогнозировщика
        
        Args:
            data: временной ряд для прогнозирования
            horizon: горизонт прогнозирования
        """
        self.data = data
        self.horizon = horizon
        self.models_results = {}
        
    def detect_seasonality(self) -> Tuple[bool, int]:
       
        # Используем ACF для поиска сезонности
        if len(self.data) < 50:
            return False, 0
            
        acf_values = acf(self.data, nlags=min(40, len(self.data)//2))
        
        # Ищем пики в ACF после лага 1
        peaks = []
        for i in range(2, len(acf_values)-1):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                if acf_values[i] > 0.3:  # Порог значимости
                    peaks.append(i)
        
        if peaks:
            # Берем первый значимый пик как период сезонности
            period = peaks[0]
            return True, period
        
        return False, 0
    
    def check_stationarity(self) -> bool:
        """
        Проверка стационарности ряда (тест Дики-Фуллера)
    
        """
        result = adfuller(self.data)
        return result[1] < 0.05  # p-value < 0.05 означает стационарность
    
    def fit_arima(self) -> Dict[str, Any]:
        """
        Подбор и обучение модели ARIMA/SARIMA
   
        """
        try:
            # Определяем сезонность
            has_seasonality, period = self.detect_seasonality()
            
            # Автоматический подбор параметров
            if has_seasonality and period > 1:
                # SARIMA для сезонных данных
                model = pm.auto_arima(
                    self.data,
                    seasonal=True,
                    m=period,
                    start_p=0, start_q=0,
                    max_p=3, max_q=3,
                    start_P=0, start_Q=0,
                    max_P=2, max_Q=2,
                    d=None, D=None,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    n_fits=50
                )
            else:
                # ARIMA для несезонных данных
                model = pm.auto_arima(
                    self.data,
                    seasonal=False,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5,
                    d=None,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
            
            forecast = model.predict(n_periods=self.horizon)
            
            fitted_values = model.predict_in_sample()
            train_mse = mean_squared_error(self.data[1:], fitted_values[1:])
            
            return {
                'model': 'ARIMA' if not has_seasonality else 'SARIMA',
                'forecast': forecast,
                'train_mse': train_mse,
                'params': model.get_params(),
                'seasonal_period': period if has_seasonality else None
            }
            
        except Exception as e:
            print(f"Ошибка в ARIMA: {e}")
            return None
    
    def fit_holt_winters(self) -> Dict[str, Any]:
        """
        Подбор и обучение модели Хольта-Винтерса
      
        """
        try:
            # Определяем сезонность
            has_seasonality, period = self.detect_seasonality()
            
            # Проверяем достаточность данных для сезонной модели
            if has_seasonality and period > 1 and len(self.data) >= 2 * period:
                # Сезонная модель
                model = ExponentialSmoothing(
                    self.data,
                    seasonal_periods=period,
                    trend='add',
                    seasonal='add',
                    initialization_method='estimated'
                )
            else:
                # Модель без сезонности
                model = ExponentialSmoothing(
                    self.data,
                    trend='add',
                    seasonal=None,
                    initialization_method='estimated'
                )
            
            fitted_model = model.fit(optimized=True)
            
            # Прогнозирование
            forecast = fitted_model.forecast(self.horizon)
            
            # Оценка качества
            fitted_values = fitted_model.fittedvalues
            train_mse = mean_squared_error(self.data, fitted_values)
            
            return {
                'model': 'Holt-Winters',
                'forecast': forecast,
                'train_mse': train_mse,
                'seasonal': has_seasonality,
                'seasonal_period': period if has_seasonality else None
            }
            
        except Exception as e:
            print(f"Ошибка в Holt-Winters: {e}")
            return None
    
    def fit_simple_models(self) -> Dict[str, Any]:
        """
        Простые базовые модели для сравнения
      
        """
        results = {}
        
        # Наивный прогноз (последнее значение)
        naive_forecast = np.repeat(self.data.iloc[-1], self.horizon)
        results['naive'] = {
            'model': 'Naive',
            'forecast': naive_forecast,
            'train_mse': 0  # Для наивного прогноза не считаем MSE
        }
        
        # Прогноз по тренду (линейная регрессия)
        try:
            x = np.arange(len(self.data))
            y = self.data.values
            
            # МНК для линейной регрессии (условия Гаусса-Маркова)
            x_mean = x.mean()
            y_mean = y.mean()
            
            beta_1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            beta_0 = y_mean - beta_1 * x_mean
            
            # Прогноз
            future_x = np.arange(len(self.data), len(self.data) + self.horizon)
            trend_forecast = beta_0 + beta_1 * future_x
            
            # Оценка качества
            fitted_values = beta_0 + beta_1 * x
            train_mse = mean_squared_error(y, fitted_values)
            
            results['trend'] = {
                'model': 'Linear Trend',
                'forecast': trend_forecast,
                'train_mse': train_mse
            }
        except:
            pass
            
        return results
    
    def ensemble_forecast(self, models_results: List[Dict]) -> np.ndarray:
        """
        Ансамблирование прогнозов разных моделей
       
        """
        if not models_results:
            return None
            
        # Фильтруем успешные модели
        valid_models = [m for m in models_results if m is not None and 'forecast' in m]
        
        if not valid_models:
            return None
            
        if len(valid_models) == 1:
            return valid_models[0]['forecast']
        
        # Вычисляем веса на основе обратного MSE
        weights = []
        for model in valid_models:
            mse = model.get('train_mse', 1e10)
            if mse == 0:
                mse = 1e-10
            weights.append(1 / mse)
        
        # Нормализуем веса
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Взвешенное среднее прогнозов
        ensemble_forecast = np.zeros(self.horizon)
        for i, model in enumerate(valid_models):
            ensemble_forecast += weights[i] * model['forecast']
        
        return ensemble_forecast
    
    def forecast(self) -> pd.DataFrame:
        """
        Основной метод прогнозирования
        Выбирает лучшую модель или ансамбль моделей
        """
        print("Анализ временного ряда...")
        print(f"Длина ряда: {len(self.data)}")
        print(f"Горизонт прогнозирования: {self.horizon}")
        
        # Обучаем разные модели
        models_results = []
        
        # ARIMA/SARIMA
        print("\nОбучение ARIMA/SARIMA...")
        arima_result = self.fit_arima()
        if arima_result:
            models_results.append(arima_result)
            print(f"ARIMA обучена, MSE: {arima_result['train_mse']:.4f}")
        
        # Holt-Winters
        print("\nОбучение Holt-Winters...")
        hw_result = self.fit_holt_winters()
        if hw_result:
            models_results.append(hw_result)
            print(f"Holt-Winters обучена, MSE: {hw_result['train_mse']:.4f}")
        
        # Простые модели
        print("\nОбучение базовых моделей...")
        simple_results = self.fit_simple_models()
        for name, result in simple_results.items():
            if result:
                models_results.append(result)
                if result['train_mse'] > 0:
                    print(f"{result['model']} обучена, MSE: {result['train_mse']:.4f}")
        
        # Выбор лучшей модели или ансамбля
        if not models_results:
            raise ValueError("Не удалось обучить ни одну модель")
        
        # Сортируем модели по MSE
        sorted_models = sorted(
            [m for m in models_results if m.get('train_mse', 0) > 0],
            key=lambda x: x['train_mse']
        )
        
        # Используем ансамбль топ-3 моделей или лучшую модель
        if len(sorted_models) >= 2:
            print("\nИспользуется ансамбль моделей")
            final_forecast = self.ensemble_forecast(sorted_models[:3])
        else:
            best_model = sorted_models[0] if sorted_models else models_results[0]
            print(f"\nИспользуется модель: {best_model['model']}")
            final_forecast = best_model['forecast']
        
        # Создаем DataFrame с результатами
        result_df = pd.DataFrame({
            'forecast': final_forecast
        })
        
        return result_df




parser = argparse.ArgumentParser(description='Прогнозирование временных рядов')
parser.add_argument('input_file', help='Путь к CSV файлу с временным рядом')
parser.add_argument('horizon', type=int, help='Горизонт прогнозирования')
parser.add_argument('--output', default='forecast.csv', help='Путь для сохранения прогноза')

args = parser.parse_args()

try:
    print(f"Загрузка данных из {args.input_file}...")
    data = pd.read_csv(args.input_file)
    
    if len(data.columns) == 1:
        ts_data = data.iloc[:, 0]
    else:
        # Если несколько колонок, берем последнюю как целевую
        ts_data = data.iloc[:, -1]
    

    forecaster = TimeSeriesForecaster(ts_data, args.horizon)
    
    # Получаем прогноз
    forecast_df = forecaster.forecast()
    

    forecast_df.to_csv(args.output, index=False, header=False)
    print(f"\nПрогноз сохранен в {args.output}")
    
except Exception as e:
    print(f"Ошибка: {e}")
    sys.exit(1)




