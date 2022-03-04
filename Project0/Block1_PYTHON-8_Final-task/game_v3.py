"""Игра угадай число
   Компьютер сам загадывает и сам угадывает число за менее чем 20 попыток
"""

import numpy as np


def random_predict(number: int=1) -> int:
    """Рандомно угадываем число 

    Args:
        number (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Число попыток
    """
    count = 0          # Число попыток
    min_number = 1
    max_number = 101
    predict_number = round((max_number - min_number)/2)  # Среднее число между макс. и мин. числами
    
    while True:
        count += 1
        if predict_number > number:
            max_number = predict_number    # Новое макс. число
        elif predict_number < number:
            min_number = predict_number    # Новое мин.число
        else:
            break # Конец игры, выход из цикла
        
        predict_number = min_number + round((max_number - min_number)/2)  # Новое сред. число
    return count


def score_game(random_predict) -> int:
    """За какое количство попыток в среднем за 1000 подходов угадывает наш алгоритм

    Args:
        random_predict ([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """
    count_ls = []
    random_array = np.random.randint(1, 101, size=(1000))  # Загадали список чисел

    for number in random_array:
        count_ls.append(random_predict(number))

    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за: {score} попыток")
    return score


if __name__ == "__main__":
    # RUN
    score_game(random_predict)