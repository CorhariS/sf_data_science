import pandas as pd
import numpy as np
import requests

from sklearn import metrics

import sys, os
sys.path.append(os.path.join(os.path.abspath(''),'..', 'libs'))



def print_metrics(y_test, y_test_predict):
    print('Valid R^2: {:.3f}'.format(metrics.r2_score(y_test, y_test_predict)))
    print('Valid MAE: {:.3f}'.format(metrics.mean_absolute_error(y_test, y_test_predict)))
    print('Valid MAPE: {:.3f}'.format(metrics.mean_absolute_percentage_error(y_test, y_test_predict)*100))

if __name__ == '__main__':
    
    X_valid = pd.read_csv('../libs/data/df_X_valid.csv', dtype={"zipcode": str})
    y_valid = pd.read_csv('../libs/data/df_y_valid.csv')    
      
    # выполняем POST-запрос на сервер по эндпоинту add с параметром json
    r = requests.post('http://localhost:5000/predict', json=X_valid.to_json())
    # выводим статус запроса
    print('Status code: {}'.format(r.status_code))
    # реализуем обработку результата
    if r.status_code == 200:
        # если запрос выполнен успешно (код обработки=200),
        y_pred = np.array(r.json()['predictions'])
        
        #Выводим результирующие метрики
        print_metrics(y_valid, y_pred)
    else:
        # если запрос завершён с кодом, отличным от 200,
        # выводим содержимое ответа
        print(r.text)