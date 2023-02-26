# Приложение определяет нарисованное пользователем число.
Программа распознаёт рукописные цифры используя библиотеку машинного обучения tensorflow. 
Пользователь с помощью мышки рисует число в одном окне программы а во втором показывается какое число увидела программа.
Для начала работы над программой на языке python нужно установить библиотеки tensorflo, opencv-python и keras.
В func.py необходимо заранее выделить память компьтера для работы приложения.
В model.py инициализируем модели и слои tensorflow и выводит модель в отдельную переменную.
main.py производиться обучение на основе 60000 руписных чисел от 0 до 9. Создаётся два набора даныых. Первый для обучения, второй для тестирования.
Файл digits_test.py инициализирует интерфейс приложения. Подключаем необходимые используемые данные из предыдущих файлов. 
Запуская digits_test.py появляется два окна. В первом мы можем нарисовать число, а во втором сразу показывается распознанное  число.
С помощью клавиши 'd' мы можем удалить нарисованное число. Клавиша 'q' закроет прилодение.
