#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mlp.py — власна реалізація багатошарового перцептрона (Роль 1).

Реалізовано самостійно без використання бібліотек навчання нейромереж:
  - Архітектура: Input(in_dim) -> Dense(h1, ReLU) -> Dense(h2, ReLU) -> Dense(out_dim, Softmax)
  - Ініціалізація ваг: He (Kaiming Normal)
  - Функція втрат: зважена крос-ентропія (Weighted Cross-Entropy) + L2-регуляризація
  - Зворотне поширення помилки (Backpropagation)
  - Оптимізатори: Adam (за замовчуванням) та SGD із моментом
"""

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class MLPClassifier:
    """Багатошаровий перцептрон із двома прихованими шарами ReLU та виходом Softmax."""

    def __init__(
        self,
        in_dim: int = 19,
        h1: int = 24,
        h2: int = 12,
        out_dim: int = 3,
        seed: int = 3001,
        optimizer: str = "adam",
    ) -> None:
        self.in_dim = int(in_dim)
        self.h1 = int(h1)
        self.h2 = int(h2)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self.optimizer = optimizer.lower()

        rng = random.Random(self.seed)
        std1 = math.sqrt(2.0 / max(1, self.in_dim))
        std2 = math.sqrt(2.0 / max(1, self.h1))
        std3 = math.sqrt(2.0 / max(1, self.h2))

        self.W1: List[List[float]] = [
            [rng.gauss(0.0, std1) for _ in range(self.in_dim)] for _ in range(self.h1)
        ]
        self.b1: List[float] = [0.0] * self.h1

        self.W2: List[List[float]] = [
            [rng.gauss(0.0, std2) for _ in range(self.h1)] for _ in range(self.h2)
        ]
        self.b2: List[float] = [0.0] * self.h2

        self.W3: List[List[float]] = [
            [rng.gauss(0.0, std3) for _ in range(self.h2)] for _ in range(self.out_dim)
        ]
        self.b3: List[float] = [0.0] * self.out_dim

        self._reset_optimizer_state()

    def _reset_optimizer_state(self) -> None:
        self.step_count = 0
        self.mW1 = [[0.0] * self.in_dim for _ in range(self.h1)]
        self.vW1 = [[0.0] * self.in_dim for _ in range(self.h1)]
        self.mb1 = [0.0] * self.h1
        self.vb1 = [0.0] * self.h1

        self.mW2 = [[0.0] * self.h1 for _ in range(self.h2)]
        self.vW2 = [[0.0] * self.h1 for _ in range(self.h2)]
        self.mb2 = [0.0] * self.h2
        self.vb2 = [0.0] * self.h2

        self.mW3 = [[0.0] * self.h2 for _ in range(self.out_dim)]
        self.vW3 = [[0.0] * self.h2 for _ in range(self.out_dim)]
        self.mb3 = [0.0] * self.out_dim
        self.vb3 = [0.0] * self.out_dim

    @staticmethod
    def _softmax(logits: Sequence[float]) -> List[float]:
        max_z = max(logits)
        exps = [math.exp(z - max_z) for z in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def forward_single(
        self, x: Sequence[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        """Прямий прохід для одного прикладу: повертає (a1, a2, probs)."""
        W1, b1, W2, b2, W3, b3 = self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
        h1, h2, out_dim = self.h1, self.h2, self.out_dim

        a1 = [0.0] * h1
        for i in range(h1):
            val = b1[i] + sum(w * xj for w, xj in zip(W1[i], x))
            if val > 0.0:
                a1[i] = val

        a2 = [0.0] * h2
        for j in range(h2):
            val = b2[j] + sum(w * a1i for w, a1i in zip(W2[j], a1))
            if val > 0.0:
                a2[j] = val

        logits = [b3[k] + sum(w * a2j for w, a2j in zip(W3[k], a2)) for k in range(out_dim)]
        probs = self._softmax(logits)
        return a1, a2, probs

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Повертає ймовірності класів [p0, p1, p2] для кожного запису."""
        return [self.forward_single(x)[2] for x in X]

    def predict(self, X: Sequence[Sequence[float]]) -> List[int]:
        """Повертає передбачені класи (0, 1 або 2) для кожного запису."""
        preds: List[int] = []
        for x in X:
            p = self.forward_single(x)[2]
            if p[0] >= p[1] and p[0] >= p[2]:
                preds.append(0)
            elif p[1] >= p[2]:
                preds.append(1)
            else:
                preds.append(2)
        return preds

    def compute_loss(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        class_weights: Sequence[float] = (1.0, 1.0, 1.0),
        l2_reg: float = 0.0,
    ) -> float:
        """Обчислює середнє значення зваженої крос-ентропії (+ L2) на вибірці."""
        if not X:
            return 0.0
        total = 0.0
        for x, yt in zip(X, y):
            probs = self.forward_single(x)[2]
            p_true = max(probs[yt], 1e-15)
            total -= class_weights[yt] * math.log(p_true)
        loss = total / len(X)
        if l2_reg > 0.0:
            sq = (
                sum(w * w for row in self.W1 for w in row)
                + sum(w * w for row in self.W2 for w in row)
                + sum(w * w for row in self.W3 for w in row)
            )
            loss += 0.5 * l2_reg * sq
        return loss

    def train_epoch(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        batch_size: int = 64,
        lr: float = 0.006,
        class_weights: Sequence[float] = (1.0, 2.0, 2.2),
        l2_reg: float = 1e-4,
        rng: Optional[random.Random] = None,
    ) -> float:
        """Виконує одну епоху міні-батч навчання (backpropagation + Adam/SGD)."""
        n = len(X)
        if n == 0:
            return 0.0

        indices = list(range(n))
        if rng is not None:
            rng.shuffle(indices)

        W1, b1, W2, b2, W3, b3 = self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
        in_dim, h1, h2, out_dim = self.in_dim, self.h1, self.h2, self.out_dim
        total_loss = 0.0

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            inv_bs = 1.0 / len(batch_idx)

            gW1 = [[0.0] * in_dim for _ in range(h1)]
            gb1 = [0.0] * h1
            gW2 = [[0.0] * h1 for _ in range(h2)]
            gb2 = [0.0] * h2
            gW3 = [[0.0] * h2 for _ in range(out_dim)]
            gb3 = [0.0] * out_dim

            for idx in batch_idx:
                x = X[idx]
                yt = y[idx]
                cw = class_weights[yt]

                # 1. Прямий прохід (Forward pass)
                a1 = [0.0] * h1
                for i in range(h1):
                    v = b1[i] + sum(w * xj for w, xj in zip(W1[i], x))
                    if v > 0.0:
                        a1[i] = v

                a2 = [0.0] * h2
                for j in range(h2):
                    v = b2[j] + sum(w * a1i for w, a1i in zip(W2[j], a1))
                    if v > 0.0:
                        a2[j] = v

                z3 = [b3[k] + sum(w * a2j for w, a2j in zip(W3[k], a2)) for k in range(out_dim)]
                max_z = max(z3)
                exps = [math.exp(z - max_z) for z in z3]
                sum_exps = sum(exps)
                probs = [e / sum_exps for e in exps]

                total_loss -= cw * math.log(max(probs[yt], 1e-15))

                # 2. Зворотний прохід (Backpropagation)
                # Вихідний шар Softmax + Cross-Entropy: dL/dz3_k = cw * (p_k - y_k)
                dz3 = [cw * (probs[k] - (1.0 if k == yt else 0.0)) for k in range(out_dim)]
                for k in range(out_dim):
                    dz3_k = dz3[k]
                    gb3[k] += dz3_k
                    row_gW3 = gW3[k]
                    for j in range(h2):
                        row_gW3[j] += dz3_k * a2[j]

                # Другий прихований шар (ReLU)
                da1 = [0.0] * h1
                for j in range(h2):
                    if a2[j] > 0.0:
                        dz2_j = sum(W3[k][j] * dz3[k] for k in range(out_dim))
                        gb2[j] += dz2_j
                        row_gW2 = gW2[j]
                        row_W2 = W2[j]
                        for i in range(h1):
                            row_gW2[i] += dz2_j * a1[i]
                            da1[i] += row_W2[i] * dz2_j

                # Перший прихований шар (ReLU)
                for i in range(h1):
                    if a1[i] > 0.0 and da1[i] != 0.0:
                        dz1_i = da1[i]
                        gb1[i] += dz1_i
                        row_gW1 = gW1[i]
                        for d_idx in range(in_dim):
                            row_gW1[d_idx] += dz1_i * x[d_idx]

            # 3. Оновлення параметрів (Adam або SGD з моментом)
            self._apply_gradients(
                gW1, gb1, gW2, gb2, gW3, gb3, inv_bs=inv_bs, lr=lr, l2_reg=l2_reg
            )

        return total_loss / n

    def _apply_gradients(
        self,
        gW1: List[List[float]],
        gb1: List[float],
        gW2: List[List[float]],
        gb2: List[float],
        gW3: List[List[float]],
        gb3: List[float],
        inv_bs: float,
        lr: float,
        l2_reg: float,
    ) -> None:
        self.step_count += 1
        W1, b1, W2, b2, W3, b3 = self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
        in_dim, h1, h2, out_dim = self.in_dim, self.h1, self.h2, self.out_dim

        if self.optimizer == "sgd":
            momentum = 0.9
            for k in range(out_dim):
                g = gb3[k] * inv_bs
                self.mb3[k] = momentum * self.mb3[k] + g
                b3[k] -= lr * self.mb3[k]
                for j in range(h2):
                    g = gW3[k][j] * inv_bs + l2_reg * W3[k][j]
                    self.mW3[k][j] = momentum * self.mW3[k][j] + g
                    W3[k][j] -= lr * self.mW3[k][j]
            for j in range(h2):
                g = gb2[j] * inv_bs
                self.mb2[j] = momentum * self.mb2[j] + g
                b2[j] -= lr * self.mb2[j]
                for i in range(h1):
                    g = gW2[j][i] * inv_bs + l2_reg * W2[j][i]
                    self.mW2[j][i] = momentum * self.mW2[j][i] + g
                    W2[j][i] -= lr * self.mW2[j][i]
            for i in range(h1):
                g = gb1[i] * inv_bs
                self.mb1[i] = momentum * self.mb1[i] + g
                b1[i] -= lr * self.mb1[i]
                for d_idx in range(in_dim):
                    g = gW1[i][d_idx] * inv_bs + l2_reg * W1[i][d_idx]
                    self.mW1[i][d_idx] = momentum * self.mW1[i][d_idx] + g
                    W1[i][d_idx] -= lr * self.mW1[i][d_idx]
            return

        # Adam optimizer
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = self.step_count
        lr_t = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)

        for k in range(out_dim):
            g = gb3[k] * inv_bs
            self.mb3[k] = beta1 * self.mb3[k] + (1.0 - beta1) * g
            self.vb3[k] = beta2 * self.vb3[k] + (1.0 - beta2) * (g * g)
            b3[k] -= lr_t * self.mb3[k] / (math.sqrt(self.vb3[k]) + eps)
            for j in range(h2):
                g = gW3[k][j] * inv_bs + l2_reg * W3[k][j]
                self.mW3[k][j] = beta1 * self.mW3[k][j] + (1.0 - beta1) * g
                self.vW3[k][j] = beta2 * self.vW3[k][j] + (1.0 - beta2) * (g * g)
                W3[k][j] -= lr_t * self.mW3[k][j] / (math.sqrt(self.vW3[k][j]) + eps)

        for j in range(h2):
            g = gb2[j] * inv_bs
            self.mb2[j] = beta1 * self.mb2[j] + (1.0 - beta1) * g
            self.vb2[j] = beta2 * self.vb2[j] + (1.0 - beta2) * (g * g)
            b2[j] -= lr_t * self.mb2[j] / (math.sqrt(self.vb2[j]) + eps)
            for i in range(h1):
                g = gW2[j][i] * inv_bs + l2_reg * W2[j][i]
                self.mW2[j][i] = beta1 * self.mW2[j][i] + (1.0 - beta1) * g
                self.vW2[j][i] = beta2 * self.vW2[j][i] + (1.0 - beta2) * (g * g)
                W2[j][i] -= lr_t * self.mW2[j][i] / (math.sqrt(self.vW2[j][i]) + eps)

        for i in range(h1):
            g = gb1[i] * inv_bs
            self.mb1[i] = beta1 * self.mb1[i] + (1.0 - beta1) * g
            self.vb1[i] = beta2 * self.vb1[i] + (1.0 - beta2) * (g * g)
            b1[i] -= lr_t * self.mb1[i] / (math.sqrt(self.vb1[i]) + eps)
            for d_idx in range(in_dim):
                g = gW1[i][d_idx] * inv_bs + l2_reg * W1[i][d_idx]
                self.mW1[i][d_idx] = beta1 * self.mW1[i][d_idx] + (1.0 - beta1) * g
                self.vW1[i][d_idx] = beta2 * self.vW1[i][d_idx] + (1.0 - beta2) * (g * g)
                W1[i][d_idx] -= lr_t * self.mW1[i][d_idx] / (math.sqrt(self.vW1[i][d_idx]) + eps)

    def copy_weights(self) -> Dict[str, Any]:
        """Створює глибоку копію поточних ваг мережі."""
        return {
            "W1": copy.deepcopy(self.W1),
            "b1": list(self.b1),
            "W2": copy.deepcopy(self.W2),
            "b2": list(self.b2),
            "W3": copy.deepcopy(self.W3),
            "b3": list(self.b3),
        }

    def load_weights(self, weights: Dict[str, Any]) -> None:
        """Відновлює ваги мережі зі словника."""
        self.W1 = copy.deepcopy(weights["W1"])
        self.b1 = list(weights["b1"])
        self.W2 = copy.deepcopy(weights["W2"])
        self.b2 = list(weights["b2"])
        self.W3 = copy.deepcopy(weights["W3"])
        self.b3 = list(weights["b3"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": {
                "in_dim": self.in_dim,
                "h1": self.h1,
                "h2": self.h2,
                "out_dim": self.out_dim,
                "activation": "ReLU",
                "output_activation": "Softmax",
                "optimizer": self.optimizer,
                "seed": self.seed,
            },
            "weights": {
                "W1": [[round(w, 8) for w in row] for row in self.W1],
                "b1": [round(b, 8) for b in self.b1],
                "W2": [[round(w, 8) for w in row] for row in self.W2],
                "b2": [round(b, 8) for b in self.b2],
                "W3": [[round(w, 8) for w in row] for row in self.W3],
                "b3": [round(b, 8) for b in self.b3],
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLPClassifier":
        arch = data["architecture"]
        model = cls(
            in_dim=arch["in_dim"],
            h1=arch["h1"],
            h2=arch["h2"],
            out_dim=arch["out_dim"],
            seed=arch.get("seed", 3001),
            optimizer=arch.get("optimizer", "adam"),
        )
        model.load_weights(data["weights"])
        return model

    def save_json(self, path: Union[str, Path]) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "MLPClassifier":
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "model" in data:
            return cls.from_dict(data["model"])
        return cls.from_dict(data)
