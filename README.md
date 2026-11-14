# SSS-Priority-Replication

Проєкт з пріоритезованої реплікації, стиснення та криптографічного захисту журналу подій (`LOG3`).

---

## Встановлення залежностей

```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Чи є загальний старт для проєкту?

Єдиного файлу на кшталт `main.py` у репозиторії **немає**.

**Головною точкою входу для наскрізного запуску всього проєкту** є модуль Ролі 5 `src.role_5.c1_runner`, який об'єднує всі 5 ролей та еталонну модель `replication.py` і виконує повний прогін по всіх сценаріях і 12 приймальних кейсах C1:

```bash
python -m src.role_5.c1_runner
```

---

## Команди запуску окремих частин та тестів

- **Наскрізний запуск (усі 12 кейсів C1 + повний пайплайн):**
  ```bash
  python -m src.role_5.c1_runner
  ```
- **Автотести (`pytest`):**
  ```bash
  pytest -v
  ```
- **Роль 1 — навчання перцептрона (перетренування моделі на `data/train.csv` та `data/valid.csv`):**
  ```bash
  python -m src.role_1.train
  ```
- **Роль 3 — експериментальні таблиці планувальника (ДП vs жадібний, кількість копій, snapshots):**
  ```bash
  python -m src.role_3.scheduler
  ```
- **Роль 4 — експерименти з ГА, Парето та атакою (`all`, `attack`, `ga`, `pareto`):**
  ```bash
  python -m src.role_4.run_experiments attack
  python -m src.role_4.run_experiments all
  ```

---

## Узагальнена структура по ролях і файлах

### Корінь репозиторію
- `replication.py` — еталонна модель реплікації на вузлах `A`, `B`, `C` (видана готова, не змінюється).
- `team_config.json` — конфігурація команди (бюджети, сценарії, пріоритети, параметри метрики `Q`).
- `TEAM_ROLES.md` — детальний опис розподілу задач та контрактів між 5 ролями.
- `requirements.txt` — залежності (`numpy`, `matplotlib`, `cryptography`, `pytest`).
- `data/` — набори даних `train.csv`, `valid.csv`, `session.csv`.
- `models/` — збережені ваги та скейлер навченого перцептрона (`best_model_basic.json`, `best_model_basic1.json`).
- `reference/` — еталонні та пошкоджені пакети (`ref_*.bin/.json`, `bad_*.bin`, `expected_failures.json`).
- `tests/test_role_4.py` — набір unit- та інтеграційних тестів.

### Роль 1 (`src/role_1/`) — Перцептрон та інженерія ознак
- `__init__.py` — публічний API (`extract_features`, `predict_classes`, `compute_event_priorities`, `encode_class_table`).
- `data_loader.py` — завантаження CSV, валідація даних і структура `LogRecord` (16 байтів).
- `features.py` — побудова часових ознак ковзного вікна (`FeatureExtractor`) і нормалізація (`StandardScaler`).
- `mlp.py` — реалізація багатошарового перцептрона на NumPy (`MLPClassifier`).
- `metrics.py` — обчислення `macro-F1` та матриці помилок.
- `train.py` — протокол навчання з кількох ініціалізацій і вибір найкращої моделі.

### Роль 2 (`src/role_2/`) — Стиснення та серіалізація записів
- `__init__.py` — експорт публічного API модуля.
- `compression.py` — серіалізація/десеріалізація 16-байтових записів, канонічний алгоритм Гаффмана та режим `NO_COMPRESSION` (`serialize_records`, `deserialize_records`, `compress`, `decompress`).

### Роль 3 (`src/role_3/`) — Планувальник реплікації
- `__init__.py` — експорт функцій планувальника.
- `scheduler.py` — розрахунок вартості/корисності (`build_costs_and_utilities`), ДП-рюкзак (`select_copies_dp`), жадібний baseline (`select_copies_greedy`), батчінг по вузлах (`schedule_batches`) та ДП для оптимальних snapshots (`optimize_checkpoints_dp`).

### Роль 4 (`src/role_4/`) — Криптографія, ГА та Модуль атаки
- `crypto.py` — `PBKDF2-HMAC-SHA256`, `AES-256-CTR`, `HMAC-SHA256` (`derive_keys`, `encrypt_and_mac`, `verify_and_decrypt`).
- `genetic.py` — генетичний алгоритм з елітизмом та порівняння з Random Search.
- `pareto.py` — багатокритеріальна оптимізація (фронт Парето за `S`, `R`, `T`).
- `fitness.py` — обгортка оцінки параметрів через наскрізний пайплайн.
- `attack.py` — ГА-атака на слабкий XOR-шифр та тест відхилення змінених байтів репліки за HMAC.
- `run_experiments.py` — скрипт запуску експериментів і збереження звітів у `results/role_4/`.

### Роль 5 (`src/role_5/`) — Контейнер `LOG3`, Відновлювач та Інтеграція
- `errors.py` — контрольовані винятки `LogError` і коди помилок (`E_SHORT`, `E_MAGIC`, `E_LENGTH`, `E_HMAC`, `E_PAYLOAD`, `E_CODE`).
- `deps.py` — адаптер підключення модулів Ролі 2 та Ролі 4.
- `container.py` — формування та розбір бінарного контейнера `LOG3` (`pack_replica`, `unpack_replica`).
- `recoverer.py` — відновлення журналу з уцілілих реплік з дедуплікацією та сортуванням за часом (`recover_journal`).
- `pipeline.py` — наскрізний пайплайн (`evaluate_pipeline`) та обчислення метрик `S`, `R`, `T`, `Q`.
- `c1_runner.py` — автоматична перевірка 12 приймальних кейсів C1.
