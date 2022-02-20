"""guess the number game"""
"""computer thinks of a number and guesses it"""

import numpy as np

def random_predict(number: int=1) -> int:
    """randomly guess a number

    Args:
        number (int, optional): hidden number. Defaults to 1.

    Returns:
        int: number of attempts
    """
    count = 0
    
    while True:
        count += 1
        predict_number = np.random.randint(1,101)
        if predict_number == number:
            break # exit loop
    return count

def score_game(random_predict) -> int:
    """За какое количество попыток в среднем из 1000 подходов угадывает наш алгоритм

    Args:
        random_predict (_type_): функция угадывания

    Returns:
        int: среднее количество попыток eufl
    """
    counter_ls = [] # список для сохранения количества попыток
    np.random.seed(1)
    random_array = np.random.randint(1, 101, size=(1000)) #загадали список чисел
    
    for number in random_array:
        counter_ls.append(random_predict(number))
    
    score = int(np.mean(counter_ls))
    print(f'Ваш алгоритм угадывает число в среднем за: {score} попыток')
    return score

if __name__ == '__main__':
    score_game(random_predict)


    




