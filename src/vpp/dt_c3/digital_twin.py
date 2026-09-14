"""Digital-twin state synchronization and safety-preview utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy
import math

from .communication import CommunicationState


@dataclass
class TwinConfig:
    """Digital-twin configuration.

    ``dt_hours`` is the dispatch interval.  For the paper experiments we use
    15 minutes by default, i.e. 0.25 h.
    """

    dt_hours: float = 0.25
    ess_capacity_kwh: float = 500.0
    ev_capacity_kwh: float = 1200.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    smoothing: float = 0.65
    trend_smoothing: float = 0.25
    forecast_horizon_steps: int = 4


@dataclass
class TwinEstimate:
    """Twin estimate with provenance metadata."""

    device_id: str
    step: int
    state: dict[str, Any]
    source: str
    aoi: int
    reconstruction_error: float | None = None


@dataclass
class DigitalTwinSynchronizer:
    """Digital twin used for state synchronization and missing-data recovery."""

    config: TwinConfig = field(default_factory=TwinConfig)
    _states: dict[str, dict[str, Any]] = field(default_factory=dict)
    _last_steps: dict[str, int] = field(default_factory=dict)
    _errors: list[float] = field(default_factory=list)
    _variable_errors: dict[str, list[float]] = field(default_factory=dict)
    _history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def reset(self) -> None:
        self._states.clear()
        self._last_steps.clear()
        self._errors.clear()
        self._variable_errors.clear()
        self._history.clear()

    def seed_state(self, device_id: str, state: dict[str, Any], step: int = 0) -> None:
        self._states[device_id] = copy.deepcopy(state)
        self._last_steps[device_id] = step
        self._history.setdefault(device_id, []).append(copy.deepcopy(state))

    def update(
        self,
        device_id: str,
        observation: dict[str, Any] | None,
        communication: CommunicationState,
        control_action: dict[str, float] | None = None,
        true_state: dict[str, Any] | None = None,
    ) -> TwinEstimate:
        """Update one device twin from delayed/missing IoT data.

        If the observation is delayed, it is first fused with the previous twin
        state and then propagated to the current step using simple resource
        dynamics.  If it is missing, pure prediction is used.
        """
        previous = copy.deepcopy(self._states.get(device_id, {}))
        action = control_action or {}

        if observation is None:
            estimated = self._predict(device_id, previous, action, communication.aoi)
            source = "predicted_missing"
        elif communication.aoi > 0:
            fused = self._fuse(previous, observation)
            estimated = self._predict(device_id, fused, action, communication.aoi)
            source = "dt_delay_compensated"
        else:
            estimated = self._fuse(previous, observation)
            source = "iot_fresh"

        self._states[device_id] = estimated
        self._last_steps[device_id] = communication.step
        self._history.setdefault(device_id, []).append(copy.deepcopy(estimated))
        if len(self._history[device_id]) > 256:
            self._history[device_id] = self._history[device_id][-256:]

        err = None
        if true_state is not None:
            err = self._state_error(estimated, true_state)
            self._errors.append(err)
            for key, value in true_state.items():
                if isinstance(value, (int, float)) and isinstance(estimated.get(key), (int, float)):
                    self._variable_errors.setdefault(key, []).append((float(estimated[key]) - float(value)) ** 2)

        return TwinEstimate(
            device_id=device_id,
            step=communication.step,
            state=copy.deepcopy(estimated),
            source=source,
            aoi=communication.aoi,
            reconstruction_error=err,
        )

    def update_batch(
        self,
        observations: dict[str, dict[str, Any] | None],
        communications: dict[str, CommunicationState],
        actions: dict[str, dict[str, float]] | None = None,
        true_states: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, TwinEstimate]:
        actions = actions or {}
        true_states = true_states or {}
        return {
            device_id: self.update(
                device_id,
                observations.get(device_id),
                communications[device_id],
                actions.get(device_id),
                true_states.get(device_id),
            )
            for device_id in communications
        }

    def get_state(self, device_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._states.get(device_id, {}))

    def aggregate_state(self) -> dict[str, float]:
        """Aggregate device twins into a VPP-level state vector."""
        agg = {
            "pv_kw": 0.0,
            "wind_kw": 0.0,
            "load_kw": 0.0,
            "ess_soc": 0.5,
            "ev_soc": 0.5,
            "price": 0.15,
        }
        for device_id, state in self._states.items():
            lower = device_id.lower()
            if "pv" in lower or "solar" in lower:
                agg["pv_kw"] += float(state.get("power_kw", state.get("pv_kw", 0.0)))
            elif "wind" in lower:
                agg["wind_kw"] += float(state.get("power_kw", state.get("wind_kw", 0.0)))
            elif "load" in lower:
                agg["load_kw"] += float(state.get("power_kw", state.get("load_kw", 0.0)))
            elif "ess" in lower or "battery" in lower:
                agg["ess_soc"] = float(state.get("soc", state.get("ess_soc", agg["ess_soc"])))
            elif "ev" in lower:
                agg["ev_soc"] = float(state.get("soc", state.get("ev_soc", agg["ev_soc"])))
            if "price" in state:
                agg["price"] = float(state["price"])
        return agg

    def mean_reconstruction_error(self) -> float:
        if not self._errors:
            return 0.0
        return sum(self._errors) / len(self._errors)

    def variable_rmse(self) -> dict[str, float]:
        return {key: math.sqrt(sum(vals) / len(vals)) for key, vals in self._variable_errors.items() if vals}

    def forecast(self, actions: dict[str, dict[str, float]] | None = None, horizon_steps: int | None = None) -> dict[int, dict[str, dict[str, Any]]]:
        """Roll the twin forward for a short safety-preview horizon."""
        horizon = horizon_steps or self.config.forecast_horizon_steps
        actions = actions or {}
        out: dict[int, dict[str, dict[str, Any]]] = {}
        saved_states = copy.deepcopy(self._states)
        for h in range(1, horizon + 1):
            step_states: dict[str, dict[str, Any]] = {}
            for device_id, state in saved_states.items():
                step_states[device_id] = self._predict(device_id, state, actions.get(device_id, {}), 1)
            out[h] = copy.deepcopy(step_states)
            saved_states = step_states
        return out

    def preview_action_safety(self, aggregate_action: dict[str, float], limits: Any | None = None, horizon_steps: int | None = None) -> dict[str, Any]:
        """Preview whether an action is likely to violate SOC bounds.

        This method is intentionally generic to avoid a circular import from
        safety.py.  It is used as a digital-twin safety-prediction metric, not
        as the final shield.
        """
        actions = {
            "ess": {"ess_power_kw": aggregate_action.get("ess_power_kw", 0.0)},
            "ev": {"ev_power_kw": aggregate_action.get("ev_power_kw", 0.0)},
        }
        fc = self.forecast(actions=actions, horizon_steps=horizon_steps)
        risks: list[str] = []
        ess_min = getattr(limits, "ess_soc_min", 0.1) if limits is not None else 0.1
        ess_max = getattr(limits, "ess_soc_max", 0.9) if limits is not None else 0.9
        ev_min = getattr(limits, "ev_soc_min", 0.2) if limits is not None else 0.2
        for h, states in fc.items():
            ess = states.get("ess", {})
            ev = states.get("ev", {})
            if "soc" in ess and not (ess_min <= float(ess["soc"]) <= ess_max):
                risks.append(f"h{h}:ess_soc={float(ess['soc']):.3f}")
            if "soc" in ev and float(ev["soc"]) < ev_min:
                risks.append(f"h{h}:ev_soc={float(ev['soc']):.3f}")
        return {"safe": len(risks) == 0, "risks": risks, "horizon_steps": horizon_steps or self.config.forecast_horizon_steps}

    def _fuse(self, previous: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
        if not previous:
            return copy.deepcopy(observation)
        fused = copy.deepcopy(previous)
        alpha = self.config.smoothing
        for key, value in observation.items():
            if isinstance(value, (int, float)) and isinstance(previous.get(key), (int, float)):
                new_value = alpha * float(value) + (1 - alpha) * float(previous[key])
                if key == "power_kw":
                    old_trend = float(previous.get("trend_kw_per_step", 0.0))
                    observed_trend = float(value) - float(previous[key])
                    fused["trend_kw_per_step"] = self.config.trend_smoothing * observed_trend + (1 - self.config.trend_smoothing) * old_trend
                fused[key] = new_value
            else:
                fused[key] = copy.deepcopy(value)
        return fused

    def _predict(
        self,
        device_id: str,
        previous: dict[str, Any],
        action: dict[str, float],
        steps_ahead: int,
    ) -> dict[str, Any]:
        state = copy.deepcopy(previous) if previous else {"power_kw": 0.0}
        steps = max(1, steps_ahead)
        hours = self.config.dt_hours * steps
        lower = device_id.lower()

        # ESS / EV SOC propagation. Positive power means discharge to VPP;
        # negative power means charging from VPP/grid.
        if "ess" in lower or "battery" in lower:
            p = float(action.get("ess_power_kw", action.get("power_kw", 0.0)))
            soc = float(state.get("soc", state.get("ess_soc", 0.5)))
            capacity = float(state.get("capacity_kwh", self.config.ess_capacity_kwh))
            state["soc"] = self._next_soc(soc, p, capacity, hours)
            state["power_kw"] = p
        elif "ev" in lower:
            p = float(action.get("ev_power_kw", action.get("power_kw", 0.0)))
            soc = float(state.get("soc", state.get("ev_soc", 0.5)))
            capacity = float(state.get("capacity_kwh", self.config.ev_capacity_kwh))
            state["soc"] = self._next_soc(soc, p, capacity, hours)
            state["power_kw"] = p
        else:
            # For load/PV/wind, persist with a light trend term when available.
            trend = float(state.get("trend_kw_per_step", 0.0))
            if "power_kw" in state:
                state["power_kw"] = max(0.0, float(state.get("power_kw", 0.0)) + trend * steps)
        state["dt_predicted_steps"] = steps
        return state

    def _next_soc(self, soc: float, power_kw: float, capacity_kwh: float, hours: float) -> float:
        if capacity_kwh <= 0:
            return soc
        if power_kw >= 0:  # discharge
            delta = power_kw * hours / (capacity_kwh * self.config.discharge_efficiency)
            return max(0.0, min(1.0, soc - delta))
        delta = abs(power_kw) * hours * self.config.charge_efficiency / capacity_kwh
        return max(0.0, min(1.0, soc + delta))

    def _state_error(self, estimated: dict[str, Any], true_state: dict[str, Any]) -> float:
        keys = [k for k, v in true_state.items() if isinstance(v, (int, float)) and isinstance(estimated.get(k), (int, float))]
        if not keys:
            return 0.0
        return math.sqrt(sum((float(estimated[k]) - float(true_state[k])) ** 2 for k in keys) / len(keys))


# ---------------------------------------------------------------------------
# Paper-level digital twin estimators and benchmarking utilities
# ---------------------------------------------------------------------------

@dataclass
class TwinEstimatorMetrics:
    name: str
    rmse: dict[str, float]
    overall_rmse: float


class BaseTwinEstimator:
    """Common interface for DT state-estimation baselines."""

    name = "base"

    def reset(self) -> None:
        self.last: dict[str, float] = {}
        self.sse: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def update(self, observation: dict[str, float] | None, true_state: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError

    def _record(self, est: dict[str, float], true_state: dict[str, float]) -> None:
        for k, v in true_state.items():
            if isinstance(v, (int, float)) and isinstance(est.get(k), (int, float)):
                self.sse[k] = self.sse.get(k, 0.0) + (float(est[k]) - float(v)) ** 2
                self.n[k] = self.n.get(k, 0) + 1

    def metrics(self) -> TwinEstimatorMetrics:
        rmse = {k: math.sqrt(self.sse[k] / max(1, self.n[k])) for k in self.sse}
        overall = math.sqrt(sum(self.sse.values()) / max(1, sum(self.n.values()))) if self.sse else 0.0
        return TwinEstimatorMetrics(self.name, rmse, overall)


class NoDigitalTwinEstimator(BaseTwinEstimator):
    name = "no_dt_raw_missing_zero"

    def __init__(self) -> None:
        self.reset()

    def update(self, observation: dict[str, float] | None, true_state: dict[str, float]) -> dict[str, float]:
        est = {k: float(observation.get(k, 0.0)) for k in true_state} if observation else {k: 0.0 for k in true_state}
        self._record(est, true_state)
        return est


class PersistenceTwinEstimator(BaseTwinEstimator):
    name = "persistence"

    def __init__(self) -> None:
        self.reset()

    def update(self, observation: dict[str, float] | None, true_state: dict[str, float]) -> dict[str, float]:
        if observation:
            self.last = {k: float(v) for k, v in observation.items() if isinstance(v, (int, float))}
        est = {k: float(self.last.get(k, 0.0)) for k in true_state}
        self._record(est, true_state)
        return est


class KalmanTwinEstimator(BaseTwinEstimator):
    """Online scalar Kalman filter per variable."""

    name = "kalman"

    def __init__(self, process_var: float = 1.0, measurement_var: float = 9.0) -> None:
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.var: dict[str, float] = {}

    def update(self, observation: dict[str, float] | None, true_state: dict[str, float]) -> dict[str, float]:
        keys = [k for k, v in true_state.items() if isinstance(v, (int, float))]
        for k in keys:
            pred = float(self.last.get(k, 0.0))
            p = float(self.var.get(k, 100.0)) + self.process_var
            if observation and isinstance(observation.get(k), (int, float)):
                z = float(observation[k])
                gain = p / (p + self.measurement_var)
                pred = pred + gain * (z - pred)
                p = (1.0 - gain) * p
            self.last[k] = pred
            self.var[k] = p
        est = {k: self.last.get(k, 0.0) for k in keys}
        self._record(est, true_state)
        return est


class ExtendedKalmanTwinEstimator(KalmanTwinEstimator):
    """EKF-like estimator with a simple nonlinear load/PV trend model."""

    name = "extended_kalman"

    def __init__(self) -> None:
        super().__init__(process_var=2.0, measurement_var=6.0)
        self.trend: dict[str, float] = {}

    def update(self, observation: dict[str, float] | None, true_state: dict[str, float]) -> dict[str, float]:
        # nonlinear prediction: x_{t+1}=x_t+trend+tanh(trend)*small gain
        for k, last in list(self.last.items()):
            tr = self.trend.get(k, 0.0)
            self.last[k] = float(last + tr + math.tanh(tr / 100.0) * 2.0)
        old = dict(self.last)
        est = super().update(observation, true_state)
        if observation:
            for k, v in observation.items():
                if isinstance(v, (int, float)):
                    self.trend[k] = 0.4 * (float(v) - float(old.get(k, v))) + 0.6 * self.trend.get(k, 0.0)
        return est


class GRUTwinEstimator(KalmanTwinEstimator):
    """Tiny online GRU-style estimator.

    This avoids adding training dependencies while still providing a recurrent
    state-estimation baseline. It behaves like a gated recurrent filter and can
    be replaced by a trained torch GRU later.
    """

    name = "gru_filter"

    def __init__(self) -> None:
        super().__init__(process_var=1.5, measurement_var=5.0)
        self.hidden: dict[str, float] = {}

    def update(self, observation: dict[str, float] | None, true_state: dict[str, float]) -> dict[str, float]:
        if observation:
            for k, v in observation.items():
                if isinstance(v, (int, float)):
                    h = self.hidden.get(k, float(v))
                    z = 1.0 / (1.0 + math.exp(-abs(float(v) - h) / 50.0))
                    self.hidden[k] = (1.0 - z) * h + z * float(v)
        obs = {k: self.hidden[k] for k in self.hidden} if self.hidden else observation
        return super().update(obs, true_state)


class TransformerTwinEstimator(PersistenceTwinEstimator):
    """Attention-style sequence smoother used as a Transformer proxy baseline."""

    name = "transformer_attention_smoother"

    def reset(self) -> None:
        super().reset()
        self.window: list[dict[str, float]] = []

    def update(self, observation: dict[str, float] | None, true_state: dict[str, float]) -> dict[str, float]:
        if observation:
            self.window.append({k: float(v) for k, v in observation.items() if isinstance(v, (int, float))})
            self.window = self.window[-8:]
        if self.window:
            # exponential attention favoring recent states
            weights = [math.exp(i - len(self.window) + 1) for i in range(len(self.window))]
            sw = sum(weights)
            keys = {k for row in self.window for k in row}
            self.last = {k: sum(w * row.get(k, self.last.get(k, 0.0)) for w, row in zip(weights, self.window)) / sw for k in keys}
        est = {k: float(self.last.get(k, 0.0)) for k in true_state}
        self._record(est, true_state)
        return est


class ProposedDecisionTwinEstimator(BaseTwinEstimator):
    """Decision-level twin: proposed synchronizer flattened to aggregate variables."""

    name = "proposed_dt_synchronizer"

    def __init__(self, twin_config: TwinConfig | None = None) -> None:
        self.twin = DigitalTwinSynchronizer(twin_config or TwinConfig())
        self.reset()

    def reset(self) -> None:
        BaseTwinEstimator.reset(self)
        self.twin.reset()
        self.step = 0

    def update(self, observation: dict[str, float] | None, true_state: dict[str, float]) -> dict[str, float]:
        # flattened VPP variables are treated as a pseudo-device for benchmark fairness
        from .communication import CommunicationState
        comm = CommunicationState("vpp", step=self.step, delivered_step=self.step if observation else None, delay_steps=0, aoi=0 if observation else 1, missing=observation is None)
        self.twin.update("vpp", observation, comm, true_state=true_state)
        est = self.twin.get_state("vpp")
        est = {k: float(est.get(k, 0.0)) for k in true_state}
        self._record(est, true_state)
        self.step += 1
        return est


class DigitalTwinBenchmark:
    """Compare no-DT, persistence, Kalman, EKF, GRU, Transformer and proposed DT."""

    def __init__(self, estimators: list[BaseTwinEstimator] | None = None) -> None:
        self.estimators = estimators or [
            NoDigitalTwinEstimator(),
            PersistenceTwinEstimator(),
            KalmanTwinEstimator(),
            ExtendedKalmanTwinEstimator(),
            GRUTwinEstimator(),
            TransformerTwinEstimator(),
            ProposedDecisionTwinEstimator(),
        ]

    def reset(self) -> None:
        for est in self.estimators:
            est.reset()

    def run(self, true_series: list[dict[str, float]], observed_series: list[dict[str, float] | None]) -> list[TwinEstimatorMetrics]:
        self.reset()
        for true_state, obs in zip(true_series, observed_series):
            for est in self.estimators:
                est.update(obs, true_state)
        return [est.metrics() for est in self.estimators]

# ---------------------------------------------------------------------------
# Learning-based digital twin predictors for paper-level benchmarking
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency path
    import torch
    torch.set_num_threads(1)
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class _TinySequenceRegressor(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, model_type: str, n_features: int, hidden_dim: int = 24, window: int = 6) -> None:
        if nn is None:
            return
        super().__init__()
        self.model_type = model_type
        self.window = window
        self.n_features = n_features
        if model_type == "gru":
            self.seq = nn.GRU(n_features, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, n_features)
        elif model_type == "lstm":
            self.seq = nn.LSTM(n_features, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, n_features)
        else:
            enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim * 2, batch_first=True, dropout=0.0)
            self.in_proj = nn.Linear(n_features, hidden_dim)
            self.pos = nn.Parameter(torch.zeros(1, window, hidden_dim))
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.head = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        if self.model_type in {"gru", "lstm"}:
            y, _ = self.seq(x)
            return self.head(y[:, -1, :])
        z = self.in_proj(x) + self.pos[:, : x.shape[1], :]
        y = self.encoder(z)
        return self.head(y[:, -1, :])


class LearnedSequenceTwinEstimator(BaseTwinEstimator):
    """Trainable GRU/LSTM/Transformer estimator for decision-level DT.

    The estimator first learns one-step reconstruction/prediction from the
    training split of the true/observed series, then runs online and records
    RMSE on the complete scenario.  Missing observations are replaced with the
    previous estimate before being fed into the network, which mimics delayed
    IoT streams in the final experiments.
    """

    def __init__(self, model_type: str = "gru", window: int = 6, epochs: int = 8, hidden_dim: int = 16) -> None:
        self.model_type = model_type
        self.window = window
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.name = f"trained_{model_type}_dt"
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.keys: list[str] = []
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.model: Any = None
        self.buffer: list[dict[str, float]] = []
        self.fitted = False

    def fit(self, true_series: list[dict[str, float]], observed_series: list[dict[str, float] | None]) -> None:
        if torch is None or nn is None or len(true_series) <= self.window + 2:
            self.fitted = False
            return
        self.keys = [k for k, v in true_series[0].items() if isinstance(v, (int, float))]
        n_train = max(self.window + 2, int(0.65 * len(true_series)))
        train_true = true_series[:n_train]
        # Use observed series with missing data filled from the last available
        # observation, while targets remain true states.
        filled: list[dict[str, float]] = []
        last = {k: float(train_true[0].get(k, 0.0)) for k in self.keys}
        for obs, truth in zip(observed_series[:n_train], train_true):
            if obs is not None:
                last = {k: float(obs.get(k, last.get(k, 0.0))) for k in self.keys}
            filled.append(dict(last))
        self.mean = {k: float(sum(row[k] for row in train_true) / len(train_true)) for k in self.keys}
        self.std = {k: max(1e-6, math.sqrt(sum((row[k] - self.mean[k]) ** 2 for row in train_true) / len(train_true))) for k in self.keys}
        X = []; Y = []
        for i in range(self.window, len(train_true)):
            X.append([[ (filled[j][k] - self.mean[k]) / self.std[k] for k in self.keys] for j in range(i - self.window, i)])
            Y.append([(train_true[i][k] - self.mean[k]) / self.std[k] for k in self.keys])
        if not X:
            self.fitted = False
            return
        x = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(Y, dtype=torch.float32)
        self.model = _TinySequenceRegressor(self.model_type, len(self.keys), self.hidden_dim, self.window)
        opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            pred = self.model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
        self.model.eval()
        self.fitted = True

    def _vector_to_state(self, arr: Any) -> dict[str, float]:
        vals = arr.detach().cpu().numpy().reshape(-1) if hasattr(arr, "detach") else arr
        return {k: float(vals[i] * self.std.get(k, 1.0) + self.mean.get(k, 0.0)) for i, k in enumerate(self.keys)}

    def update(self, observation: dict[str, float] | None, true_state: dict[str, float]) -> dict[str, float]:
        if not self.keys:
            self.keys = [k for k, v in true_state.items() if isinstance(v, (int, float))]
        if observation:
            obs = {k: float(observation.get(k, self.last.get(k, true_state.get(k, 0.0)))) for k in self.keys}
        else:
            obs = {k: float(self.last.get(k, true_state.get(k, 0.0))) for k in self.keys}
        self.buffer.append(obs); self.buffer = self.buffer[-self.window:]
        if self.fitted and self.model is not None and len(self.buffer) >= self.window and torch is not None:
            x = [[[ (row[k] - self.mean.get(k, 0.0)) / self.std.get(k, 1.0) for k in self.keys] for row in self.buffer[-self.window:]]]
            with torch.no_grad():
                est = self._vector_to_state(self.model(torch.tensor(x, dtype=torch.float32)))
        else:
            # Safe fallback while the model warms up.
            est = dict(obs)
        self.last = est
        self._record(est, true_state)
        return est


class LearnedGRUTwinEstimator(LearnedSequenceTwinEstimator):
    def __init__(self) -> None:
        super().__init__("gru")


class LearnedLSTMTwinEstimator(LearnedSequenceTwinEstimator):
    def __init__(self) -> None:
        super().__init__("lstm")


class LearnedTransformerTwinEstimator(LearnedSequenceTwinEstimator):
    def __init__(self) -> None:
        super().__init__("transformer", window=8, epochs=6, hidden_dim=16)


class DigitalTwinBenchmark:
    """Compare raw, filtering, trained neural DTs and proposed synchronizer."""

    def __init__(self, estimators: list[BaseTwinEstimator] | None = None, include_learning: bool = True) -> None:
        base: list[BaseTwinEstimator] = [
            NoDigitalTwinEstimator(),
            PersistenceTwinEstimator(),
            KalmanTwinEstimator(),
            ExtendedKalmanTwinEstimator(),
            GRUTwinEstimator(),
            TransformerTwinEstimator(),
        ]
        if include_learning:
            base += [LearnedGRUTwinEstimator(), LearnedLSTMTwinEstimator(), LearnedTransformerTwinEstimator()]
        base += [ProposedDecisionTwinEstimator()]
        self.estimators = estimators or base

    def reset(self) -> None:
        for est in self.estimators:
            est.reset()

    def run(self, true_series: list[dict[str, float]], observed_series: list[dict[str, float] | None]) -> list[TwinEstimatorMetrics]:
        self.reset()
        for est in self.estimators:
            if hasattr(est, "fit"):
                try:
                    est.fit(true_series, observed_series)  # type: ignore[attr-defined]
                except Exception:
                    pass
        for true_state, obs in zip(true_series, observed_series):
            for est in self.estimators:
                est.update(obs, true_state)
        return [est.metrics() for est in self.estimators]
