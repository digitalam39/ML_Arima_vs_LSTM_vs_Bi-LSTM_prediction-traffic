
# Базовые библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Метрики и препроцессинг
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Статистические модели
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# TensorFlow / Keras
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Воспроизводимость
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Стиль графиков
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.titleweight"] = "bold"

print("TensorFlow:", tf.__version__)
print("Окружение готово ✓")

def generate_iot_dataset(n_samples=50000, num_days=90, random_state=42):
    """Генерация синтетического датасета IoT-сети (расширенная версия из ЛР1)."""
    np.random.seed(random_state)

    device_profiles = {
        "Smart_Camera":         {"packet_size": (500, 1500), "packet_rate": (20, 50),
                                 "bandwidth": (2, 8),        "protocol": ["TCP", "UDP"],
                                 "port": [80, 443, 554],     "session_duration": (300, 3600),
                                 "latency": (10, 50),        "jitter": (2, 10),    "packet_loss": (0, 2)},
        "Smart_Thermostat":     {"packet_size": (50, 200),   "packet_rate": (1, 5),
                                 "bandwidth": (0.01, 0.1),   "protocol": ["MQTT", "CoAP"],
                                 "port": [1883, 5683],       "session_duration": (60, 600),
                                 "latency": (20, 100),       "jitter": (5, 20),    "packet_loss": (0, 1)},
        "Smart_Lock":           {"packet_size": (100, 300),  "packet_rate": (0.1, 2),
                                 "bandwidth": (0.001, 0.05), "protocol": ["MQTT", "HTTP"],
                                 "port": [1883, 443],        "session_duration": (1, 30),
                                 "latency": (30, 150),       "jitter": (10, 30),   "packet_loss": (0, 1)},
        "Smart_Light":          {"packet_size": (50, 150),   "packet_rate": (0.5, 3),
                                 "bandwidth": (0.005, 0.05), "protocol": ["MQTT", "CoAP"],
                                 "port": [1883, 5683],       "session_duration": (10, 300),
                                 "latency": (15, 80),        "jitter": (3, 15),    "packet_loss": (0, 1)},
        "Motion_Sensor":        {"packet_size": (50, 100),   "packet_rate": (0.1, 1),
                                 "bandwidth": (0.001, 0.01), "protocol": ["MQTT", "CoAP"],
                                 "port": [1883, 5683],       "session_duration": (1, 10),
                                 "latency": (20, 100),       "jitter": (5, 20),    "packet_loss": (0, 1)},
        "Smart_Speaker":        {"packet_size": (200, 800),  "packet_rate": (10, 30),
                                 "bandwidth": (0.5, 3),      "protocol": ["TCP", "UDP"],
                                 "port": [443, 80],          "session_duration": (60, 1800),
                                 "latency": (20, 80),        "jitter": (5, 20),    "packet_loss": (0, 2)},
        "Smart_TV":             {"packet_size": (800, 1500), "packet_rate": (30, 60),
                                 "bandwidth": (5, 25),       "protocol": ["TCP", "HTTP"],
                                 "port": [80, 443],          "session_duration": (1800, 7200),
                                 "latency": (15, 60),        "jitter": (3, 15),    "packet_loss": (0, 3)},
        "Fitness_Tracker":      {"packet_size": (100, 300),  "packet_rate": (0.1, 2),
                                 "bandwidth": (0.01, 0.1),   "protocol": ["HTTP", "MQTT"],
                                 "port": [443, 1883],        "session_duration": (30, 300),
                                 "latency": (30, 150),       "jitter": (10, 30),   "packet_loss": (0, 2)},
        "Smart_Refrigerator":   {"packet_size": (100, 400),  "packet_rate": (1, 5),
                                 "bandwidth": (0.05, 0.5),   "protocol": ["HTTP", "MQTT"],
                                 "port": [443, 1883],        "session_duration": (60, 600),
                                 "latency": (25, 120),       "jitter": (8, 25),    "packet_loss": (0, 1)},
        "Environmental_Sensor": {"packet_size": (50, 150),   "packet_rate": (0.1, 1),
                                 "bandwidth": (0.001, 0.02), "protocol": ["MQTT", "CoAP"],
                                 "port": [1883, 5683],       "session_duration": (5, 60),
                                 "latency": (25, 120),       "jitter": (8, 25),    "packet_loss": (0, 1)},
    }

    devices = list(device_profiles.keys())
    samples_per_device = n_samples // len(devices)
    start_time = datetime(2025, 1, 1)
    total_seconds = num_days * 24 * 3600

    data = []
    for device_type in devices:
        profile = device_profiles[device_type]
        for _ in range(samples_per_device):
            # Случайная временная метка
            ts = start_time + timedelta(seconds=int(np.random.randint(0, total_seconds)))
            hour = ts.hour

            # Суточная сезонность: трафик активнее днём (особенно 9–22)
            day_factor = 0.6 + 0.8 * np.exp(-0.5 * ((hour - 14) / 6) ** 2)

            packet_size      = np.random.uniform(*profile["packet_size"])
            packet_rate      = np.random.uniform(*profile["packet_rate"]) * day_factor
            bandwidth        = np.random.uniform(*profile["bandwidth"]) * day_factor
            protocol         = np.random.choice(profile["protocol"])
            port             = np.random.choice(profile["port"])
            session_duration = np.random.uniform(*profile["session_duration"])
            latency          = np.random.uniform(*profile["latency"])
            jitter           = np.random.uniform(*profile["jitter"])
            packet_loss      = np.random.uniform(*profile["packet_loss"])

            inter_arrival_time = 1000 / packet_rate if packet_rate > 0 else 1000
            burst_rate         = packet_rate * np.random.uniform(1.0, 3.0)
            connection_freq    = 3600 / session_duration if session_duration > 0 else 1
            data_volume        = (packet_size * packet_rate * session_duration) / (1024 * 1024)

            is_anomaly = 1 if np.random.random() < 0.05 else 0
            if is_anomaly:
                packet_size *= np.random.uniform(2, 5)
                packet_rate *= np.random.uniform(3, 10)
                latency     *= np.random.uniform(2, 5)
                packet_loss *= np.random.uniform(5, 20)
                bandwidth   *= np.random.uniform(2, 5)

            data.append({
                "timestamp": ts, "device_type": device_type,
                "packet_size": packet_size, "packet_rate": packet_rate,
                "bandwidth_usage": bandwidth, "protocol": protocol, "port": port,
                "session_duration": session_duration, "inter_arrival_time": inter_arrival_time,
                "jitter": jitter, "packet_loss": packet_loss, "latency": latency,
                "hour_of_day": ts.hour, "day_of_week": ts.weekday(),
                "is_weekend": int(ts.weekday() >= 5),
                "burst_rate": burst_rate, "connection_frequency": connection_freq,
                "data_volume": data_volume, "is_anomaly": is_anomaly,
            })

    df = pd.DataFrame(data).sort_values("timestamp").reset_index(drop=True)
    return df


print("Генерация IoT-датасета (90 дней, 50 000 записей)...")
df = generate_iot_dataset(n_samples=50000, num_days=90, random_state=SEED)
df.to_csv("iot_network_data.csv", index=False)
print(f"Готово: {df.shape[0]} строк × {df.shape[1]} столбцов")
df.head()


# Краткий обзор датасета
print("Период:", df["timestamp"].min(), "→", df["timestamp"].max())
print("Уникальных устройств:", df["device_type"].nunique())
print("Доля аномалий: {:.2%}".format(df["is_anomaly"].mean()))
df.describe(include="all").T.head(15)

# Установим timestamp как индекс
df = df.set_index("timestamp").sort_index()

# Часовой агрегат: суммарная полоса пропускания по всей сети
ts = df["bandwidth_usage"].resample("1h").sum().rename("traffic_mbps")

# Также сохраним вспомогательные ряды для EDA
ts_packets = df["packet_rate"].resample("1h").sum().rename("packets_per_sec")
ts_volume  = df["data_volume"].resample("1h").sum().rename("data_volume_mb")

print(f"Длина временного ряда: {len(ts)} часовых отсчётов")
print(f"Период: {ts.index.min()} → {ts.index.max()}")
print(ts.describe())
ts.head()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(ts.index, ts.values, linewidth=0.8, color="navy")
ax.set_title("Суммарный трафик IoT-сети (часовая агрегация)", fontsize=14)
ax.set_xlabel("Время")
ax.set_ylabel("bandwidth_usage, Mbps (сумма по сети)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("01_timeseries_overview.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Среднее значение по часам суток
hourly = ts.groupby(ts.index.hour).mean()
axes[0].plot(hourly.index, hourly.values, marker="o", color="darkblue", linewidth=2)
axes[0].fill_between(hourly.index, hourly.values, alpha=0.2, color="skyblue")
axes[0].set_title("Среднее по часам суток")
axes[0].set_xlabel("Час суток")
axes[0].set_ylabel("Средний трафик, Mbps")
axes[0].set_xticks(range(0, 24, 2))
axes[0].grid(alpha=0.3)

# Среднее по дням недели
days_ru = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
weekly = ts.groupby(ts.index.weekday).mean()
colors = ["steelblue"] * 5 + ["crimson"] * 2
axes[1].bar(range(7), weekly.values, color=colors, edgecolor="black")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(days_ru)
axes[1].set_title("Среднее по дням недели")
axes[1].set_ylabel("Средний трафик, Mbps")
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("02_seasonality.png", dpi=150, bbox_inches="tight")
plt.show()

heatmap_data = ts.groupby([ts.index.weekday, ts.index.hour]).mean().unstack()

plt.figure(figsize=(16, 5))
sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={"label": "Средний трафик, Mbps"})
plt.yticks(np.arange(7) + 0.5, days_ru, rotation=0)
plt.title("Тепловая карта трафика (день недели × час суток)")
plt.xlabel("Час суток")
plt.ylabel("День недели")
plt.tight_layout()
plt.savefig("03_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

decomposition = seasonal_decompose(ts, model="additive", period=24)

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
decomposition.observed.plot(ax=axes[0], color="navy", linewidth=0.8)
axes[0].set_title("Исходный ряд")
decomposition.trend.plot(ax=axes[1], color="darkgreen")
axes[1].set_title("Тренд")
decomposition.seasonal.plot(ax=axes[2], color="darkorange")
axes[2].set_title("Сезонная компонента (период = 24 часа)")
decomposition.resid.plot(ax=axes[3], color="firebrick", linewidth=0.6)
axes[3].set_title("Остатки")
for a in axes:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("04_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()

def adf_report(series, name="ряд"):
    result = adfuller(series.dropna())
    print(f"ADF-тест для «{name}»:")
    print(f"  ADF статистика: {result[0]:.4f}")
    print(f"  p-value:        {result[1]:.4f}")
    print(f"  Критические значения: " + ", ".join(f"{k}={v:.3f}" for k, v in result[4].items()))
    if result[1] < 0.05:
        print("  → Ряд стационарен (отвергаем H0).")
    else:
        print("  → Ряд НЕстационарен (не отвергаем H0).")
    print()

adf_report(ts, "исходный ряд")
adf_report(ts.diff().dropna(), "1-е разности")

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
plot_acf(ts.dropna(),  ax=axes[0, 0], lags=48, title="ACF — исходный ряд")
plot_pacf(ts.dropna(), ax=axes[0, 1], lags=48, title="PACF — исходный ряд", method="ywm")
plot_acf(ts.diff().dropna(),  ax=axes[1, 0], lags=48, title="ACF — 1-е разности")
plot_pacf(ts.diff().dropna(), ax=axes[1, 1], lags=48, title="PACF — 1-е разности", method="ywm")
plt.tight_layout()
plt.savefig("05_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()

# 6.1 Пропуски
print("Пропусков до:", ts.isna().sum())
ts = ts.asfreq("1h").interpolate(method="linear")
print("Пропусков после интерполяции:", ts.isna().sum())

# 6.2 Мягкая обрезка экстремальных значений (выше 99-го перцентиля)
upper = ts.quantile(0.99)
n_clipped = (ts > upper).sum()
ts_clean = ts.clip(upper=upper)
print(f"Обрезано {n_clipped} значений выше 99-го перцентиля ({upper:.2f} Mbps)")

# 6.3 Разбиение по времени: 70% / 15% / 15%
n = len(ts_clean)
i_train = int(n * 0.70)
i_val   = int(n * 0.85)

train = ts_clean.iloc[:i_train]
val   = ts_clean.iloc[i_train:i_val]
test  = ts_clean.iloc[i_val:]

print(f"Train: {len(train)}  ({train.index.min().date()} → {train.index.max().date()})")
print(f"Val:   {len(val)}  ({val.index.min().date()} → {val.index.max().date()})")
print(f"Test:  {len(test)}  ({test.index.min().date()} → {test.index.max().date()})")

# Визуализация разбиения
plt.figure(figsize=(16, 4))
plt.plot(train.index, train.values, label="Train", color="steelblue", linewidth=0.8)
plt.plot(val.index,   val.values,   label="Val",   color="orange",    linewidth=0.8)
plt.plot(test.index,  test.values,  label="Test",  color="crimson",   linewidth=0.8)
plt.title("Разбиение временного ряда на train / val / test")
plt.ylabel("Трафик, Mbps")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("06_train_val_test_split.png", dpi=150, bbox_inches="tight")
plt.show()

# 6.4 Масштабирование (фитим только на train!)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
val_scaled   = scaler.transform(val.values.reshape(-1, 1)).flatten()
test_scaled  = scaler.transform(test.values.reshape(-1, 1)).flatten()
print("Train scaled diapason:", train_scaled.min(), train_scaled.max())

# 6.5 Формирование скользящих окон для LSTM
LOOKBACK = 24   # «история» — 24 часа
HORIZON  = 6    # прогноз — до 6 часов вперёд

def make_windows(series, lookback, horizon):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y)

# Для непрерывности соединим train→val→test и нарежем окна,
# но границы будем уважать через индексирование
X_train, y_train = make_windows(train_scaled, LOOKBACK, HORIZON)
X_val,   y_val   = make_windows(val_scaled,   LOOKBACK, HORIZON)
X_test,  y_test  = make_windows(test_scaled,  LOOKBACK, HORIZON)

# Под Keras требуется форма (samples, timesteps, features)
X_train = X_train[..., np.newaxis]
X_val   = X_val[...,   np.newaxis]
X_test  = X_test[...,  np.newaxis]

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def evaluate_per_step(y_true, y_pred, model_name):
    """Возвращает DataFrame: метрики для каждого шага горизонта."""
    rows = []
    for h in range(y_true.shape[1]):
        yt = y_true[:, h]
        yp = y_pred[:, h]
        rows.append({
            "model": model_name,
            "horizon_h": h + 1,
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "MAE":  mean_absolute_error(yt, yp),
            "MAPE": mape(yt, yp),
        })
    return pd.DataFrame(rows)

# Подбор порядков ARIMA на трейне (с учётом суточной сезонности m=24)
print("Подбор порядков auto_arima (это займёт ~1 минуту)...")
arima_model = pm.auto_arima(
    train.values,
    seasonal=True, m=24,
    start_p=1, start_q=1, max_p=3, max_q=3,
    d=None, D=1, max_D=1,
    start_P=0, start_Q=0, max_P=2, max_Q=2,
    information_criterion="aic",
    stepwise=True, suppress_warnings=True,
    error_action="ignore", trace=False,
)
print(arima_model.summary())
print("Найденные порядки:", arima_model.order, arima_model.seasonal_order)

# Rolling forecast на тестовой выборке
# На каждом шаге: обучаем модель на (train + val + всё, что уже видели в test),
# прогнозируем HORIZON шагов вперёд, сравниваем с реальным.
# Для скорости НЕ переобучаем с нуля — используем update().

print("Rolling forecast ARIMA...")
history = np.concatenate([train.values, val.values])
arima_running = pm.ARIMA(order=arima_model.order,
                        seasonal_order=arima_model.seasonal_order,
                        suppress_warnings=True)
arima_running.fit(history)

arima_preds = []
arima_truth = []
test_values = test.values

# Сколько окон будет в тесте при HORIZON=6 и шаге 1
n_windows = len(test_values) - HORIZON + 1
for i in range(n_windows):
    forecast = arima_running.predict(n_periods=HORIZON)
    arima_preds.append(forecast)
    arima_truth.append(test_values[i:i + HORIZON])
    # Обновляем модель одним новым наблюдением
    arima_running.update(test_values[i:i + 1])
    if (i + 1) % 50 == 0:
        print(f"  обработано окон: {i + 1}/{n_windows}")

arima_preds = np.array(arima_preds)
arima_truth = np.array(arima_truth)
print("Готово. Форма прогнозов:", arima_preds.shape)

# Метрики ARIMA по горизонту
arima_metrics = evaluate_per_step(arima_truth, arima_preds, "ARIMA")
print(arima_metrics.to_string(index=False))

def build_lstm(lookback, horizon):
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(horizon),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model

lstm_model = build_lstm(LOOKBACK, HORIZON)
lstm_model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

history_lstm = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1,
)

# Кривая обучения LSTM
plt.figure(figsize=(12, 5))
plt.plot(history_lstm.history["loss"],     label="train loss", linewidth=2)
plt.plot(history_lstm.history["val_loss"], label="val loss",   linewidth=2)
plt.title("Кривая обучения LSTM")
plt.xlabel("Эпоха")
plt.ylabel("MSE loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("07_lstm_learning_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# Прогноз LSTM на тесте + обратное масштабирование
lstm_preds_scaled = lstm_model.predict(X_test, verbose=0)
lstm_preds = scaler.inverse_transform(lstm_preds_scaled)
lstm_truth = scaler.inverse_transform(y_test)

# Метрики
lstm_metrics = evaluate_per_step(lstm_truth, lstm_preds, "LSTM")
print(lstm_metrics.to_string(index=False))

def build_bilstm(lookback, horizon):
    model = Sequential([
        Input(shape=(lookback, 1)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(horizon),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model

bilstm_model = build_bilstm(LOOKBACK, HORIZON)
bilstm_model.summary()

early_stop_bi = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

history_bilstm = bilstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    callbacks=[early_stop_bi],
    verbose=1,
)

# Кривая обучения Bi-LSTM
plt.figure(figsize=(12, 5))
plt.plot(history_bilstm.history["loss"],     label="train loss", linewidth=2)
plt.plot(history_bilstm.history["val_loss"], label="val loss",   linewidth=2)
plt.title("Кривая обучения Bi-LSTM")
plt.xlabel("Эпоха")
plt.ylabel("MSE loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("08_bilstm_learning_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# Прогноз Bi-LSTM на тесте
bilstm_preds_scaled = bilstm_model.predict(X_test, verbose=0)
bilstm_preds = scaler.inverse_transform(bilstm_preds_scaled)
bilstm_truth = scaler.inverse_transform(y_test)

bilstm_metrics = evaluate_per_step(bilstm_truth, bilstm_preds, "Bi-LSTM")
print(bilstm_metrics.to_string(index=False))

# Сводная таблица
all_metrics = pd.concat([arima_metrics, lstm_metrics, bilstm_metrics], ignore_index=True)
print("Метрики по каждому шагу горизонта:")
print(all_metrics.to_string(index=False))

# Средние метрики по всему горизонту
mean_metrics = all_metrics.groupby("model")[["RMSE", "MAE", "MAPE"]].mean().round(3)
print("\nСредние метрики по горизонту 1–6 ч:")
print(mean_metrics)

mean_metrics.to_csv("metrics_summary.csv")
all_metrics.to_csv("metrics_per_horizon.csv", index=False)

# Барплоты метрик
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric in zip(axes, ["RMSE", "MAE", "MAPE"]):
    pivot = all_metrics.pivot(index="horizon_h", columns="model", values=metric)
    pivot = pivot[["ARIMA", "LSTM", "Bi-LSTM"]]
    pivot.plot(kind="bar", ax=ax, color=["#888888", "#1f77b4", "#d62728"], edgecolor="black")
    ax.set_title(f"{metric} по горизонту")
    ax.set_xlabel("Горизонт прогноза, ч")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("09_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Сравнение прогнозов с реальным рядом (для шагов h=1 и h=6)
def plot_forecast_comparison(truth, preds_dict, step, n_show=200, save_as=None):
    plt.figure(figsize=(16, 5))
    plt.plot(truth[:n_show, step - 1], label="Реальный трафик", color="black", linewidth=1.8)
    colors = {"ARIMA": "#888888", "LSTM": "#1f77b4", "Bi-LSTM": "#d62728"}
    for name, preds in preds_dict.items():
        plt.plot(preds[:n_show, step - 1], label=name, color=colors[name],
                 linewidth=1.2, alpha=0.85)
    plt.title(f"Прогноз на {step} час(а/ов) вперёд (первые {n_show} окон теста)")
    plt.xlabel("Окно теста")
    plt.ylabel("Трафик, Mbps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.show()

preds_dict = {"ARIMA": arima_preds, "LSTM": lstm_preds, "Bi-LSTM": bilstm_preds}
truth_ref = lstm_truth   # реальные значения одинаковы у всех (одинаковые окна)

plot_forecast_comparison(truth_ref, preds_dict, step=1, save_as="10_forecast_h1.png")
plot_forecast_comparison(truth_ref, preds_dict, step=6, save_as="11_forecast_h6.png")

# Какая модель лучшая по интегральной метрике?
ranking = mean_metrics.sort_values("RMSE")
print("Рейтинг моделей по среднему RMSE на горизонте 1–6 ч:")
print(ranking)
print(f"\nЛучшая модель: {ranking.index[0]}")

