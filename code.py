# ============================================================
# Quantifying the Cost of Interpretability:
# XAI for IoT Intrusion Detection on Edge Devices
#
# Dataset: CICIoT2023
# Models: XGBoost, CNN
# XAI Methods: TreeSHAP, KernelSHAP, GradientSHAP, LIME
# ============================================================

import subprocess, sys, os

def _install(pkg):
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["datasets", "xgboost", "shap", "scikit-learn",
            "tensorflow", "psutil", "scipy"]:
    _install(pkg)

try:
    subprocess.run(["apt-get", "install", "-y", "-q", "cpulimit"],
                   check=False, capture_output=True)
except Exception:
    pass

import gc, time, json, warnings, platform, tracemalloc
import numpy as np
import pandas as pd
import psutil
warnings.filterwarnings("ignore")

from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from scipy import stats
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
import shap

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# EXPERIMENT_MODE: "GPU", "CPU", or "CPU_LIMITED"
# ============================================================
EXPERIMENT_MODE = "GPU"
# ============================================================

CONFIG = {
    "test_size": 0.15,
    "n_warmup": 10,
    "n_baseline_iters": 100,
    "n_xai_iters": 100,
    "cnn_epochs": 15,
    "cnn_batch_size": 128,
    "cnn_background_size": 20,
    "n_perturb_lime": 100,
    "n_lime_samples_xgb": 10,
    "n_lime_samples_cnn": 3,
    "n_kernel_samples": 10,
}

RPI4_TDP_WATTS = 5.0
CPU_LIMIT_PERCENT = 25

CACHE_DIR = "./cache_ciciot"
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_FILES = {
    "X_train": f"{CACHE_DIR}/X_train.npy",
    "X_test":  f"{CACHE_DIR}/X_test.npy",
    "X_val":   f"{CACHE_DIR}/X_val.npy",
    "y_train": f"{CACHE_DIR}/y_train.npy",
    "y_test":  f"{CACHE_DIR}/y_test.npy",
    "y_val":   f"{CACHE_DIR}/y_val.npy",
    "xgb_model": f"{CACHE_DIR}/xgb_model.json",
    "cnn_model": f"{CACHE_DIR}/cnn_model.keras",
}


def apply_resource_limits(mode):
    if mode != "CPU_LIMITED":
        print(f"Mode: {mode} - no resource limits")
        return None
    print("Applying resource limits to emulate Raspberry Pi 4:")
    try:
        original_cores = os.cpu_count()
        os.sched_setaffinity(0, {0, 1})
        print(f"  Cores: {original_cores} -> 2")
    except Exception as e:
        print(f"  Core limit failed: {e}")
    cpulimit_proc = None
    try:
        pid = os.getpid()
        result = subprocess.run(["which", "cpulimit"], capture_output=True, text=True)
        if result.returncode == 0:
            cpulimit_proc = subprocess.Popen(
                ["cpulimit", "-l", str(CPU_LIMIT_PERCENT), "-p", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"  cpulimit: {CPU_LIMIT_PERCENT}% (PID: {pid})")
            time.sleep(2)
    except Exception as e:
        print(f"  cpulimit failed: {e}")
    return cpulimit_proc


_cpulimit_proc = apply_resource_limits(EXPERIMENT_MODE)


def detect_environment():
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "gpu": gpus[0].name if gpus else "None",
        "mode": EXPERIMENT_MODE,
        "cpu_limit_pct": CPU_LIMIT_PERCENT if EXPERIMENT_MODE == "CPU_LIMITED" else 100,
    }


def load_ciciot2023():
    print("[1] Loading CICIoT2023 from Hugging Face...")
    ds = load_dataset("lacg030175/CIC-IoT-2023", "random_3way")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    val_df = ds["validation"].to_pandas()
    del ds
    gc.collect()
    print(f" Train: {train_df.shape}")
    print(f" Test: {test_df.shape}")
    print(f" Val: {val_df.shape}")
    return train_df, test_df, val_df


def preprocess_ciciot(df, scaler=None):
    df = df.copy()
    y = df["label"].values.astype(np.int32)
    exclude = {"label", "Label", "attack_class", "Label_orig"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, y, scaler


def train_xgboost(X_train, y_train, use_gpu):
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": SEED,
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": 1 if EXPERIMENT_MODE == "CPU_LIMITED" else -1,
        "tree_method": "hist",
    }
    if use_gpu and EXPERIMENT_MODE == "GPU":
        params["device"] = "cuda"
    start = time.perf_counter()
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return model, elapsed


def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(128, 3, activation="relu"),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(X_train, y_train):
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    model = build_cnn((X_train.shape[1], 1))
    start = time.perf_counter()
    model.fit(
        X_train_cnn, y_train,
        epochs=CONFIG["cnn_epochs"],
        batch_size=CONFIG["cnn_batch_size"],
        validation_split=0.15,
        verbose=1,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    )
    elapsed = time.perf_counter() - start
    return model, elapsed


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
    }


def measure_latency_cpu(fn, X_data, n_samples=50, n_warmup=10, confidence=0.95):
    for _ in range(n_warmup):
        try:
            _ = fn(X_data[:1])
        except Exception as e:
            return (np.nan, np.nan, np.nan, np.nan, 0, str(e))
    latencies = []
    n = min(n_samples, len(X_data))
    for i in range(n):
        try:
            cpu_start = time.process_time()
            _ = fn(X_data[i:i + 1])
            cpu_end = time.process_time()
            latencies.append((cpu_end - cpu_start) * 1000.0)
        except Exception as e:
            return (np.nan, np.nan, np.nan, np.nan, 0, str(e))
    arr = np.array(latencies)
    if len(arr) > 10:
        cutoff = np.percentile(arr, 90)
        arr_clean = arr[arr <= cutoff]
    else:
        arr_clean = arr
    mean = float(np.mean(arr_clean))
    std = float(np.std(arr_clean, ddof=1)) if len(arr_clean) > 1 else 0.0
    if len(arr_clean) > 1:
        se = std / np.sqrt(len(arr_clean))
        t_crit = stats.t.ppf((1 + confidence) / 2, df=len(arr_clean) - 1)
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
    else:
        ci_low = ci_high = mean
    return (mean, std, float(ci_low), float(ci_high), len(arr_clean), None)


def measure_memory(fn, x, n_warmup=3):
    try:
        for _ in range(n_warmup):
            _ = fn(x)
        gc.collect()
        tracemalloc.start()
        _ = fn(x)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {"peak_mb": peak / (1024 * 1024)}
    except Exception:
        return {"peak_mb": np.nan}


def energy_per_1000(latency_ms, tdp_watts=RPI4_TDP_WATTS):
    if np.isnan(latency_ms):
        return np.nan
    seconds = (latency_ms / 1000.0) * 1000
    return seconds * tdp_watts


def make_predictors(xgb_model, cnn_model):
    def xgb_predict(x):
        return xgb_model.predict_proba(x)
    def cnn_predict(x):
        x_r = x.reshape(x.shape[0], x.shape[1], 1)
        return cnn_model.predict(x_r, verbose=0)
    def cnn_forward(x):
        x_r = x.reshape(x.shape[0], x.shape[1], 1)
        return cnn_model(x_r, training=False).numpy()
    return xgb_predict, cnn_predict, cnn_forward


def lime_cpu_time(model_predict, X_data, n_samples=10, n_perturb=100):
    latencies = []
    n = min(n_samples, len(X_data))
    for i in range(n):
        x = X_data[i:i + 1]
        cpu_start = time.process_time()
        noise = np.random.normal(0, 0.1, (n_perturb, x.shape[1]))
        perturbed = np.vstack([x] * n_perturb) + noise
        preds = np.asarray(model_predict(perturbed))
        if preds.ndim == 2 and preds.shape[1] == 2:
            preds = preds[:, 1]
        else:
            preds = preds.flatten()
        if len(preds) != n_perturb:
            preds = preds[:n_perturb]
        ridge = Ridge(alpha=1.0)
        ridge.fit(perturbed, preds)
        cpu_end = time.process_time()
        latencies.append((cpu_end - cpu_start) * 1000.0)
    arr = np.array(latencies)
    return (
        float(np.mean(arr)),
        float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        len(arr),
    )


def cache_exists():
    return all(os.path.exists(p) for p in CACHE_FILES.values())


def save_cache(X_train, X_test, X_val, y_train, y_test, y_val, xgb_model, cnn_model):
    print("[CACHE] Saving...")
    np.save(CACHE_FILES["X_train"], X_train)
    np.save(CACHE_FILES["X_test"], X_test)
    np.save(CACHE_FILES["X_val"], X_val)
    np.save(CACHE_FILES["y_train"], y_train)
    np.save(CACHE_FILES["y_test"], y_test)
    np.save(CACHE_FILES["y_val"], y_val)
    xgb_model.save_model(CACHE_FILES["xgb_model"])
    cnn_model.save(CACHE_FILES["cnn_model"])
    print("[CACHE] Saved.")


def load_cache():
    print("[CACHE] Loading...")
    X_train = np.load(CACHE_FILES["X_train"])
    X_test = np.load(CACHE_FILES["X_test"])
    X_val = np.load(CACHE_FILES["X_val"])
    y_train = np.load(CACHE_FILES["y_train"])
    y_test = np.load(CACHE_FILES["y_test"])
    y_val = np.load(CACHE_FILES["y_val"])
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(CACHE_FILES["xgb_model"])
    cnn_model = load_model(CACHE_FILES["cnn_model"])
    print("[CACHE] Loaded.")
    return (X_train, X_test, X_val, y_train, y_test, y_val, xgb_model, cnn_model)


def norm_shap(sv, n_feat):
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[0, :, 1]
    elif sv.ndim == 2:
        sv = sv[0]
    return np.abs(sv).flatten()[:n_feat]


def deletion_auc(fn, X, model, feature_means, steps=10):
    aucs = []
    for i in range(len(X)):
        x = X[i:i + 1]
        base = model.predict_proba(x)[0, 1]
        imp = norm_shap(fn(x), x.shape[1])
        idx_sorted = np.argsort(imp)[::-1]
        preds = [base]
        n_feat = len(idx_sorted)
        step_size = max(1, n_feat // steps)
        for k in range(step_size, n_feat + 1, step_size):
            to_del = idx_sorted[:k]
            xp = x.copy()
            xp[0, to_del] = feature_means[to_del]
            preds.append(model.predict_proba(xp)[0, 1])
        xs = np.linspace(0, 1, len(preds))
        aucs.append(np.trapz(preds, xs))
    return float(np.mean(aucs)), float(np.std(aucs))


def stability(fn, X, k=5):
    scores = []
    for i in range(len(X)):
        x = X[i:i + 1]
        dists = np.linalg.norm(X - x, axis=1)
        nb = np.argsort(dists)[1:k + 1]
        base = norm_shap(fn(x), x.shape[1])
        if np.sum(base) == 0:
            continue
        for n in nb:
            ne = norm_shap(fn(X[n:n + 1]), x.shape[1])
            if np.sum(ne) > 0:
                scores.append(cosine_similarity([base], [ne])[0, 0])
    return float(np.mean(scores)), float(np.std(scores))


def sparsity(fn, X, thresh=0.9):
    counts = []
    for i in range(len(X)):
        imp = norm_shap(fn(X[i:i + 1]), X.shape[1])
        imp = np.sort(imp)[::-1]
        t = np.sum(imp)
        if t == 0:
            continue
        cs = np.cumsum(imp) / t
        counts.append(int(np.argmax(cs >= thresh) + 1))
    return float(np.mean(counts)), float(np.std(counts))


def main():
    env = detect_environment()
    print("=" * 70)
    print(f"ENVIRONMENT - MODE: {EXPERIMENT_MODE}")
    print("=" * 70)
    for k, v in env.items():
        print(f" {k:<15}: {v}")
    print("=" * 70)

    use_gpu = env["gpu"] != "None" and EXPERIMENT_MODE == "GPU"
    xgb_train_time = 0.0
    cnn_train_time = 0.0

    if cache_exists():
        (X_train, X_test, X_val, y_train, y_test, y_val,
         xgb_model, cnn_model) = load_cache()
    else:
        train_df, test_df, val_df = load_ciciot2023()
        print("\n[2] Preprocessing...")
        X_train, y_train, scaler = preprocess_ciciot(train_df)
        X_test, y_test, _ = preprocess_ciciot(test_df, scaler)
        X_val, y_val, _ = preprocess_ciciot(val_df, scaler)
        del train_df, test_df, val_df
        gc.collect()
        print(f" Train: {X_train.shape}, Test: {X_test.shape}")
        print("\n[3] Training XGBoost...")
        xgb_model, xgb_train_time = train_xgboost(X_train, y_train, use_gpu)
        print(f" Done in {xgb_train_time:.2f}s")
        print("\n[4] Training CNN...")
        cnn_model, cnn_train_time = train_cnn(X_train, y_train)
        print(f" Done in {cnn_train_time:.2f}s ({cnn_model.count_params():,} params)")
        save_cache(X_train, X_test, X_val, y_train, y_test, y_val, xgb_model, cnn_model)

    print("\n[5] Detection performance...")
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    cnn_proba = cnn_model.predict(X_test_cnn, verbose=0).flatten()
    cnn_pred = (cnn_proba > 0.5).astype(int)

    xgb_metrics = compute_metrics(y_test, xgb_pred, xgb_proba)
    cnn_metrics = compute_metrics(y_test, cnn_pred, cnn_proba)

    print(f"\n {'Model':<10} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'AUC':<10}")
    print(" " + "-" * 58)
    for name, m in [("XGBoost", xgb_metrics), ("CNN", cnn_metrics)]:
        print(f" {name:<10} "
              f"{m['accuracy'] * 100:>8.2f}% "
              f"{m['precision'] * 100:>8.2f}% "
              f"{m['recall'] * 100:>8.2f}% "
              f"{m['f1'] * 100:>8.2f}% "
              f"{m['roc_auc'] * 100:>8.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, xgb_pred)
    print(f"\n Confusion Matrix (XGBoost):")
    print(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    xgb_predict, cnn_predict, cnn_forward = make_predictors(xgb_model, cnn_model)

    print("\n[6] Baseline (CPU time)...")
    xgb_base = measure_latency_cpu(xgb_predict, X_test,
                                   CONFIG["n_baseline_iters"], CONFIG["n_warmup"])
    cnn_forward_base = measure_latency_cpu(cnn_forward, X_test,
                                           CONFIG["n_baseline_iters"], CONFIG["n_warmup"])
    print(f" XGBoost: {xgb_base[0]:.3f} +/- {xgb_base[1]:.3f} ms CI: [{xgb_base[2]:.2f}, {xgb_base[3]:.2f}]")
    print(f" CNN: {cnn_forward_base[0]:.3f} +/- {cnn_forward_base[1]:.3f} ms CI: [{cnn_forward_base[2]:.2f}, {cnn_forward_base[3]:.2f}]")

    print("\n[7] SHAP (CPU time)...")
    tree_explainer = shap.TreeExplainer(xgb_model)
    tree_fn = lambda x: tree_explainer.shap_values(x)
    xgb_tree = measure_latency_cpu(tree_fn, X_test,
                                   CONFIG["n_xai_iters"], CONFIG["n_warmup"])
    print(f" TreeSHAP: {xgb_tree[0]:.3f} +/- {xgb_tree[1]:.3f} ms CI: [{xgb_tree[2]:.2f}, {xgb_tree[3]:.2f}]")

    background = X_train[:50]
    kernel_explainer = shap.KernelExplainer(xgb_model.predict_proba, background)
    kernel_fn = lambda x: kernel_explainer.shap_values(x, nsamples=100)
    xgb_kernel = measure_latency_cpu(kernel_fn, X_test,
                                     CONFIG["n_kernel_samples"], 2)
    print(f" KernelSHAP: {xgb_kernel[0]:.3f} +/- {xgb_kernel[1]:.3f} ms CI: [{xgb_kernel[2]:.2f}, {xgb_kernel[3]:.2f}]")

    bg_cnn = X_train[:CONFIG["cnn_background_size"]].reshape(
        CONFIG["cnn_background_size"], X_train.shape[1], 1)
    grad_explainer = shap.GradientExplainer(cnn_model, bg_cnn)

    def grad_fn(x):
        x_r = x.reshape(x.shape[0], x.shape[1], 1)
        sv = grad_explainer.shap_values(x_r)
        if isinstance(sv, list):
            return [np.asarray(s) for s in sv]
        return np.asarray(sv)

    cnn_grad = measure_latency_cpu(grad_fn, X_test,
                                   CONFIG["n_xai_iters"], CONFIG["n_warmup"])
    print(f" GradientSHAP: {cnn_grad[0]:.3f} +/- {cnn_grad[1]:.3f} ms CI: [{cnn_grad[2]:.2f}, {cnn_grad[3]:.2f}]")

    print("\n[8] LIME (CPU time)...")
    r = lime_cpu_time(xgb_predict, X_test,
                      CONFIG["n_lime_samples_xgb"], CONFIG["n_perturb_lime"])
    xgb_lime = (r[0], r[1], np.nan, np.nan, r[2], None)
    print(f" XGBoost + LIME: {r[0]:.3f} +/- {r[1]:.3f} ms n={r[2]}")

    r = lime_cpu_time(cnn_predict, X_test,
                      CONFIG["n_lime_samples_cnn"], CONFIG["n_perturb_lime"])
    cnn_lime = (r[0], r[1], np.nan, np.nan, r[2], None)
    print(f" CNN + LIME: {r[0]:.3f} +/- {r[1]:.3f} ms n={r[2]}")

    print("\n[9] Memory (tracemalloc)...")
    memory = {
        "XGBoost_baseline": measure_memory(xgb_predict, X_test[:1]),
        "CNN_baseline": measure_memory(cnn_predict, X_test[:1]),
        "XGBoost_TreeSHAP": measure_memory(
            lambda x: tree_explainer.shap_values(x), X_test[:1]),
        "CNN_GradientSHAP": measure_memory(grad_fn, X_test[:1]),
    }
    print(f" {'Configuration':<25} {'Peak Memory (MB)'}")
    print(" " + "-" * 45)
    for name, m in memory.items():
        print(f" {name:<25} {m['peak_mb']:.3f}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    def rel_inc(xai, base):
        if np.isnan(xai) or np.isnan(base) or base == 0:
            return None
        return ((xai - base) / base) * 100

    rows = [
        ("XGBoost Baseline", xgb_base, None),
        ("XGBoost + TreeSHAP", xgb_tree, xgb_base[0]),
        ("XGBoost + KernelSHAP", xgb_kernel, xgb_base[0]),
        ("XGBoost + LIME", xgb_lime, xgb_base[0]),
        ("CNN Baseline", cnn_forward_base, None),
        ("CNN + GradientSHAP", cnn_grad, cnn_forward_base[0]),
        ("CNN + LIME", cnn_lime, cnn_forward_base[0]),
    ]

    print(f"\n {'Configuration':<28} {'Latency (ms)':<22} {'95% CI':<25} {'Increase'}")
    print(" " + "-" * 100)
    for name, result, base in rows:
        mean, std, ci_l, ci_h, n, _ = result
        if np.isnan(mean):
            print(f" {name:<28} {'FAILED':<22} N/A")
            continue
        inc = rel_inc(mean, base) if base else None
        inc_str = f"+{inc:.1f}%" if inc is not None else "---"
        ci_str = f"[{ci_l:.2f}, {ci_h:.2f}]" if not np.isnan(ci_l) else "N/A"
        print(f" {name:<28} "
              f"{f'{mean:.3f} +/- {std:.3f}':<22} "
              f"{ci_str:<25} {inc_str}")

    print("\n Memory (peak MB):")
    for name, m in memory.items():
        print(f" {name:<25} {m['peak_mb']:.3f}")

    print("\n Energy per 1000 predictions (Joules, RPi4 proxy):")
    for name, result, _ in rows:
        e = energy_per_1000(result[0])
        if np.isnan(e):
            print(f" {name:<28} FAILED")
        else:
            print(f" {name:<28} {e:.2f} J")

    # ============================================================
    # Explanation Quality Metrics
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPLANATION QUALITY METRICS")
    print("=" * 70)

    FEATURE_MEANS = X_train.mean(axis=0)

    print("\n[10] TreeSHAP quality...")
    Xe = X_test[:100]
    da_t = deletion_auc(tree_fn, Xe, xgb_model, FEATURE_MEANS)
    s_t = stability(tree_fn, Xe)
    sp_t = sparsity(tree_fn, Xe)
    print(f"  Deletion AUC: {da_t[0]:.4f} +/- {da_t[1]:.4f}")
    print(f"  Stability:    {s_t[0]:.4f} +/- {s_t[1]:.4f}")
    print(f"  Sparsity:     {sp_t[0]:.2f}")

    print("\n[11] KernelSHAP quality...")
    def kernel_fn_q(x):
        r = kernel_explainer.shap_values(x, nsamples=100)
        if isinstance(r, list): r = r[0]
        r = np.array(r)
        if r.ndim == 3: r = r[:, :, 1]
        return r
    Xk = X_test[:20]
    da_k = deletion_auc(kernel_fn_q, Xk, xgb_model, FEATURE_MEANS)
    s_k = stability(kernel_fn_q, Xk)
    sp_k = sparsity(kernel_fn_q, Xk)
    print(f"  Deletion AUC: {da_k[0]:.4f} +/- {da_k[1]:.4f}")
    print(f"  Stability:    {s_k[0]:.4f} +/- {s_k[1]:.4f}")
    print(f"  Sparsity:     {sp_k[0]:.2f}")

    print("\n[12] LIME quality...")
    def lime_fn_q(x):
        noise = np.random.normal(0, 0.1, (100, x.shape[1]))
        pert = np.vstack([x] * 100) + noise
        preds = xgb_model.predict_proba(pert)[:, 1]
        ridge = Ridge(alpha=1.0)
        ridge.fit(pert, preds)
        return np.abs(ridge.coef_).reshape(1, -1)
    da_l = deletion_auc(lime_fn_q, Xk, xgb_model, FEATURE_MEANS)
    s_l = stability(lime_fn_q, Xk)
    sp_l = sparsity(lime_fn_q, Xk)
    print(f"  Deletion AUC: {da_l[0]:.4f} +/- {da_l[1]:.4f}")
    print(f"  Stability:    {s_l[0]:.4f} +/- {s_l[1]:.4f}")
    print(f"  Sparsity:     {sp_l[0]:.2f}")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        "experiment_mode": EXPERIMENT_MODE,
        "environment": env,
        "config": CONFIG,
        "dataset": "CICIoT2023",
        "detection": {
            "XGBoost": xgb_metrics,
            "CNN": cnn_metrics,
        },
        "confusion_matrix": {
            "TN": int(cm[0,0]),
            "FP": int(cm[0,1]),
            "FN": int(cm[1,0]),
            "TP": int(cm[1,1]),
        },
        "training_time_s": {
            "XGBoost": xgb_train_time,
            "CNN": cnn_train_time,
        },
        "latency_ms": {
            "XGBoost_baseline":    {"mean": xgb_base[0], "std": xgb_base[1], "ci95": [xgb_base[2], xgb_base[3]]},
            "XGBoost_TreeSHAP":    {"mean": xgb_tree[0], "std": xgb_tree[1], "ci95": [xgb_tree[2], xgb_tree[3]]},
            "XGBoost_KernelSHAP":  {"mean": xgb_kernel[0], "std": xgb_kernel[1], "ci95": [xgb_kernel[2], xgb_kernel[3]]},
            "XGBoost_LIME":        {"mean": xgb_lime[0], "std": xgb_lime[1]},
            "CNN_baseline_forward":{"mean": cnn_forward_base[0], "std": cnn_forward_base[1], "ci95": [cnn_forward_base[2], cnn_forward_base[3]]},
            "CNN_GradientSHAP":    {"mean": cnn_grad[0], "std": cnn_grad[1], "ci95": [cnn_grad[2], cnn_grad[3]]},
            "CNN_LIME":            {"mean": cnn_lime[0], "std": cnn_lime[1]},
        },
        "memory_peak_mb": {name: m["peak_mb"] for name, m in memory.items()},
        "energy_proxy_J_per_1k": {
            name: energy_per_1000(result[0]) for name, result, _ in rows
        },
        "explanation_quality": {
            "TreeSHAP":   {"deletion_auc": da_t, "stability": s_t, "sparsity": sp_t},
            "KernelSHAP": {"deletion_auc": da_k, "stability": s_k, "sparsity": sp_k},
            "LIME":       {"deletion_auc": da_l, "stability": s_l, "sparsity": sp_l},
        },
    }

    def to_serializable(obj):
        if isinstance(obj, dict):    return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, tuple):   return [to_serializable(x) for x in obj]
        return obj

    out_json = f"results_ciciot2023_{EXPERIMENT_MODE}.json"
    with open(out_json, "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    print(f"\n[OK] Saved: {out_json}")
    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETED - MODE: {EXPERIMENT_MODE}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    _ = main()
