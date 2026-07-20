from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topoworld.scenediffuserpp.schema import AgentType
from topoworld.scenediffuserpp.schema import LightState


POSITION_SCALE = 1000.0
SIZE_MEAN = np.array([4.5, 2.0, 1.75], dtype=np.float32)
SIZE_STD = np.array([2.5, 0.8, 0.6], dtype=np.float32)
AGENT_CONTINUOUS_DIM = 7
AGENT_TYPE_COUNT = len(AgentType)
LIGHT_STATE_COUNT = len(LightState)


def wrap_to_pi(value):
    wrapped = (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def _signed_one_hot(index: int, count: int) -> np.ndarray:
    if index < 0 or index >= count:
        raise ValueError(f"one-hot index {index} is outside [0, {count})")
    result = np.full(count, -1.0, dtype=np.float32)
    result[index] = 1.0
    return result


@dataclass(frozen=True)
class AgentNormalizer:
    def encode_continuous(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float32)
        if raw.shape != (AGENT_CONTINUOUS_DIM,):
            raise ValueError("agent continuous values must have shape (7,)")
        encoded = np.empty_like(raw)
        encoded[:3] = raw[:3] / POSITION_SCALE
        encoded[3] = wrap_to_pi(raw[3]) / np.pi
        encoded[4:7] = (raw[4:7] - SIZE_MEAN) / (2.0 * SIZE_STD)
        return encoded

    def decode_continuous(self, values: np.ndarray) -> np.ndarray:
        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape != (AGENT_CONTINUOUS_DIM,):
            raise ValueError("encoded agent continuous values must have shape (7,)")
        decoded = np.empty_like(encoded)
        decoded[:3] = encoded[:3] * POSITION_SCALE
        decoded[3] = wrap_to_pi(encoded[3] * np.pi)
        decoded[4:7] = encoded[4:7] * (2.0 * SIZE_STD) + SIZE_MEAN
        return decoded

    def encode_agent(
        self,
        continuous: np.ndarray,
        type_index: AgentType | int,
        valid: bool,
    ) -> np.ndarray:
        result = np.zeros(12, dtype=np.float32)
        result[-1] = 1.0 if valid else -1.0
        if not valid:
            return result
        result[:7] = self.encode_continuous(continuous)
        result[7:11] = _signed_one_hot(int(type_index), AGENT_TYPE_COUNT)
        return result

    def decode_agent(self, values: np.ndarray) -> tuple[np.ndarray, AgentType, bool]:
        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape != (12,):
            raise ValueError("encoded agent must have shape (12,)")
        valid = bool(encoded[-1] >= 0.0)
        return (
            self.decode_continuous(encoded[:7]),
            AgentType(int(np.argmax(encoded[7:11]))),
            valid,
        )


@dataclass(frozen=True)
class LightNormalizer:
    def encode_light(
        self,
        xyz: np.ndarray,
        state: LightState | int,
        valid: bool,
    ) -> np.ndarray:
        result = np.zeros(13, dtype=np.float32)
        result[-1] = 1.0 if valid else -1.0
        if not valid:
            return result
        position = np.asarray(xyz, dtype=np.float32)
        if position.shape != (3,):
            raise ValueError("traffic-light position must have shape (3,)")
        result[:3] = position / POSITION_SCALE
        result[3:12] = _signed_one_hot(int(state), LIGHT_STATE_COUNT)
        return result

    def decode_light(self, values: np.ndarray) -> tuple[np.ndarray, LightState, bool]:
        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape != (13,):
            raise ValueError("encoded traffic light must have shape (13,)")
        return (
            encoded[:3] * POSITION_SCALE,
            LightState(int(np.argmax(encoded[3:12]))),
            bool(encoded[-1] >= 0.0),
        )

