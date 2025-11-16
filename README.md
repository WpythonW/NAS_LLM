# LLM-Driven Hyperparameter Optimization для Informer

Автоматическая оптимизация гиперпараметров модели Informer с использованием LLM (Gemini).

## Быстрый старт

```bash
# Установка зависимостей
uv sync

# Настройка API ключа
export GEMINI_API_KEY=your_api_key_here

# Запуск оптимизации
uv run optimize
```

## Colab пример
[Jupyter Notebook с примером оптимизации](https://colab.research.google.com/drive/1zLzTEc_KplZxM_XAheb7YOIYxHtFXiP4?usp=sharing)

## Структура проекта

### `config.py`
Конфигурация эксперимента и параметры оптимизации.

**Параметры GRID** (пространство поиска):
- `seq_len`: [24, 48, 96, 168, 336, 720] - длина входной последовательности
- `label_len`: [24, 48, 96, 168, 336, 720] - длина начальной части decoder
- `e_layers`: [2, 3, 4, 6] - количество encoder слоев
- `n_heads`: [8, 16] - количество attention головок
- `factor`: [3, 5, 8, 10] - фактор прореживания ProbSparse attention

**Параметры FIXED** (фиксированные):
- `d_model`: 512 - размерность модели
- `d_ff`: 2048 - размерность feedforward сети
- `d_layers`: 2 - количество decoder слоев
- `dropout`: 0.05 - коэффициент dropout

**Другие настройки**:
- `PRED_LENS`: [24, 48, 168, 336, 720] - горизонты прогноза (часы)
- `JOURNAL_DIR`: './experiments' - директория для логов
- `GEMINI_API_KEY` - API ключ (из `.env` файла)

---

### `llm_opt_toolkit/llm_requests.py`

#### `call_llm(prompt: str) -> List[InformerConfig]`
Запрашивает у Gemini Flash Lite батч конфигураций.

**Параметры**:
- `prompt` - текстовое задание с ограничениями и историей экспериментов

**Возвращает**: список из 5 конфигураций `InformerConfig`

**Детали**:
- Использует structured output (JSON schema)
- Thinking budget = 100 токенов
- Парсит ответ в список датаклассов

---

### `llm_opt_toolkit/optimization_journal.py`

#### `class Journal`
Логирование и анализ экспериментов.

**Методы**:

`__init__(filename=None)` - создает/загружает JSON файл с результатами

`add(config, mse, mae, trial)` - добавляет результат эксперимента
- Валидирует `label_len < seq_len`
- Конвертирует numpy типы в Python типы
- Сохраняет timestamp

`get_history_table(last_n=1500) -> str` - возвращает Markdown таблицу последних экспериментов

`count_trials() -> int` - количество завершенных экспериментов

`get_best()` - возвращает строку с лучшим MSE

`print_best()` - выводит лучшую конфигурацию в консоль

---

### `llm_opt_toolkit/prompting.py`

#### `build_prompt(pred_len, batch_size, history_md) -> str`
Генерирует промпт для LLM.

**Параметры**:
- `pred_len` - горизонт прогноза
- `batch_size` - количество конфигураций в батче
- `history_md` - Markdown таблица предыдущих экспериментов

**Включает**:
- Описание датасета (ETTh1)
- Пространство поиска
- Ограничения (label_len < seq_len, seq_len >= pred_len и т.д.)
- Историю экспериментов
- Требуемый формат JSON ответа

---

### `llm_opt_toolkit/train.py`

#### `train_informer(dataset_path, config, pred_len) -> (mse, mae)`
Обучает модель Informer с заданной конфигурацией.

**Параметры**:
- `dataset_path` - путь к CSV файлу
- `config` - объект `InformerConfig`
- `pred_len` - горизонт прогноза

**Возвращает**: кортеж (MSE, MAE) на валидации

**Настройки обучения**:
- 6 эпох, patience=3
- batch_size=32, lr=0.0001
- Автоопределение GPU/CPU
- ProbSparse attention, TimeF embedding

---

### `llm_opt_toolkit/optimizator.py`

#### `run_experiment(dataset_path, pred_len, journal_name, n_batches=3, batch_size=5)`
Основной цикл оптимизации.

**Параметры**:
- `dataset_path` - путь к данным
- `pred_len` - горизонт прогноза
- `journal_name` - имя файла журнала
- `n_batches` - количество батчей
- `batch_size` - размер батча (конфигураций от LLM)

**Процесс**:
1. Каждые `batch_size` экспериментов запрашивает новый батч у LLM
2. Обучает модель с каждой конфигурацией
3. Логирует результаты в журнал
4. Выводит лучшую конфигурацию по завершении

#### `print_comparison_table(results)`
Выводит сравнительную таблицу результатов для разных `pred_len`.


## Анализ полученных результатов

- pred_len (параметр модели) - горизонт прогнозирования для модели

1. Создать 4 json для каждой `eth_i` -> отпавить на анализ полученные таблицы
   1. Выгрузить все гипотизы для **лучших** экспериментов для всех `pred-len` (9 самых мин по скору)
   2. Выгрузить все гипотизы для **худших** экспериментов для всех `pred-len` (9 самых макс по скору)

2. Построить графики для каждого `pred-len` и `eth_i`
   1. `LLM eth1-grid12`
   2. `LLM eth1-grid3`
   3. `Optuna eth1-grid12`

3. PS -> модель прогнозируем N кол-во возможных конфигураций не дублируя предыдущие (для экономии лимитов) -> кол-во трейлов = batch_size * n_batches
4. Оформить в ридмишке