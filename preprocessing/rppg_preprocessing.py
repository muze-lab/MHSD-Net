from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


GRID_SIZE = 8
WINDOW_SIZE = 256
WINDOW_STRIDE = 10
FLOW_SCALE = 5.0


def face_box_from_68_landmarks(
    landmarks: np.ndarray,
    frame_height: int,
    frame_width: int,
) -> tuple[int, int, int, int]:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (68, 2):
        raise ValueError(f"Expected 68x2 landmarks, got {points.shape}.")

    left_x = float(points[0, 0])
    right_x = float(points[16, 0])
    chin_y = float(points[8, 1])
    brow_center_y = float(points[17:27, 1].mean())

    width = max(int(round(right_x - left_x)), 1)
    height = max(int(round(1.2 * (chin_y - brow_center_y))), 1)
    center_x = 0.5 * (left_x + right_x)
    x = int(round(center_x - width / 2))
    y = int(round(chin_y - height))

    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = max(1, min(width, frame_width - x))
    height = max(1, min(height, frame_height - y))
    return x, y, width, height


def crop_face(frame_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    return np.asarray(frame_rgb)[y : y + height, x : x + width].copy()


def _validate_frames(face_frames_rgb: np.ndarray) -> np.ndarray:
    frames = np.asarray(face_frames_rgb)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected TxHxWx3 frames, got {frames.shape}.")
    if frames.shape[0] < 2 or frames.shape[1] < GRID_SIZE or frames.shape[2] < GRID_SIZE:
        raise ValueError(f"Invalid frame sequence shape: {frames.shape}.")
    return np.clip(frames, 0, 255).astype(np.uint8)


def rgb_to_yuv(frame_rgb: np.ndarray) -> np.ndarray:
    matrix = np.asarray(
        [
            [0.299, 0.587, 0.114],
            [-0.169, -0.331, 0.500],
            [0.500, -0.419, -0.081],
        ],
        dtype=np.float32,
    )
    yuv = np.einsum("hwc,dc->hwd", np.asarray(frame_rgb, dtype=np.float32), matrix)
    yuv[..., 1:] += 128.0
    return np.clip(yuv, 0.0, 255.0)


def extract_grid_yuv(frame_rgb: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    frame = rgb_to_yuv(frame_rgb)
    height, width = frame.shape[:2]
    y_edges = np.linspace(0, height, grid_size + 1).round().astype(np.int32)
    x_edges = np.linspace(0, width, grid_size + 1).round().astype(np.int32)
    features = np.empty((grid_size * grid_size, 3), dtype=np.float32)
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            patch = frame[
                y_edges[row] : y_edges[row + 1],
                x_edges[col] : x_edges[col + 1],
            ]
            features[index] = patch.mean(axis=(0, 1))
            index += 1
    return features


def build_stmap(face_frames_rgb: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    frames = _validate_frames(face_frames_rgb)
    return np.stack(
        [extract_grid_yuv(frame, grid_size) for frame in frames],
        axis=1,
    )


def scale_stmap_window(stmap: np.ndarray) -> np.ndarray:
    data = np.asarray(stmap, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected NxTxC STMap, got {data.shape}.")
    minimum = data.min(axis=1, keepdims=True)
    maximum = data.max(axis=1, keepdims=True)
    scale = maximum - minimum
    return np.where(
        scale > 1e-8,
        (data - minimum) / np.maximum(scale, 1e-8),
        0.0,
    ).astype(np.float32)


def _motion_track(length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros((length,), dtype=np.float32)
    positions = np.linspace(-3.0, 3.0, length, dtype=np.float64)
    velocity = np.exp(-0.5 * positions**2)
    track = np.cumsum(velocity)
    track -= track[0]
    track /= track[-1]
    return track.astype(np.float32)


@dataclass(frozen=True)
class SlidingCropConfig:
    crop_ratio: float = 0.5
    working_size: int = 256
    fps: float = 30.0
    minimum_segment_seconds: float = 0.33
    maximum_segment_seconds: float = 1.5
    motion_probability: float = 0.5


def _key_times(
    frame_count: int,
    rng: np.random.Generator,
    config: SlidingCropConfig,
) -> list[int]:
    if config.fps <= 0:
        raise ValueError("fps must be positive.")
    if not 0 < config.minimum_segment_seconds <= config.maximum_segment_seconds:
        raise ValueError("Invalid segment duration range.")

    minimum_frames = max(1, int(round(config.minimum_segment_seconds * config.fps)))
    maximum_frames = max(minimum_frames, int(round(config.maximum_segment_seconds * config.fps)))
    times = [0]
    remaining = frame_count

    while remaining > maximum_frames:
        upper = min(maximum_frames, remaining - minimum_frames)
        duration = int(rng.integers(minimum_frames, upper + 1))
        times.append(times[-1] + duration)
        remaining -= duration

    times.append(frame_count)
    return times


def sliding_crop_motion_augmentation(
    face_frames_rgb: np.ndarray,
    rng: np.random.Generator,
    config: SlidingCropConfig = SlidingCropConfig(),
) -> np.ndarray:
    frames = _validate_frames(face_frames_rgb)
    frame_count, original_height, original_width, _ = frames.shape
    if not 0.0 < config.crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in (0, 1].")

    resized = np.stack(
        [
            cv2.resize(
                frame,
                (config.working_size, config.working_size),
                interpolation=cv2.INTER_CUBIC,
            )
            for frame in frames
        ],
        axis=0,
    )

    crop_size = max(1, int(config.crop_ratio * config.working_size))
    max_anchor = config.working_size - crop_size
    times = _key_times(frame_count, rng, config)

    y = int(rng.integers(0, max_anchor + 1))
    x = int(rng.integers(0, max_anchor + 1))
    anchors = [(y, x)]
    for _ in range(len(times) - 1):
        if float(rng.random()) < config.motion_probability:
            y = int(rng.integers(0, max_anchor + 1))
            x = int(rng.integers(0, max_anchor + 1))
        anchors.append((y, x))

    output = np.empty_like(frames)
    for segment in range(len(times) - 1):
        start = times[segment]
        end = times[segment + 1]
        track = _motion_track(end - start)
        start_y, start_x = anchors[segment]
        end_y, end_x = anchors[segment + 1]

        for offset in range(end - start):
            y = int(round(start_y + (end_y - start_y) * float(track[offset])))
            x = int(round(start_x + (end_x - start_x) * float(track[offset])))
            crop = resized[start + offset, y : y + crop_size, x : x + crop_size]
            output[start + offset] = cv2.resize(
                crop,
                (original_width, original_height),
                interpolation=cv2.INTER_CUBIC,
            )
    return output


def dis_flow_magnitude(
    face_frames_rgb: np.ndarray,
    flow_scale: float = FLOW_SCALE,
) -> np.ndarray:
    frames = _validate_frames(face_frames_rgb)
    if flow_scale <= 0:
        raise ValueError("flow_scale must be positive.")
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames])
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    magnitude = np.zeros(gray.shape, dtype=np.float32)
    for index in range(1, len(gray)):
        flow = estimator.calc(gray[index - 1], gray[index], None)
        magnitude[index] = np.linalg.norm(flow, axis=-1) / flow_scale * 255.0
    return magnitude


def build_flow_stmap(
    flow_magnitude: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    flow = np.asarray(flow_magnitude, dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError(f"Expected TxHxW flow maps, got {flow.shape}.")
    frame_count, height, width = flow.shape
    y_edges = np.linspace(0, height, grid_size + 1).round().astype(np.int32)
    x_edges = np.linspace(0, width, grid_size + 1).round().astype(np.int32)
    result = np.empty((grid_size * grid_size, frame_count), dtype=np.float32)
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            patch = flow[
                :,
                y_edges[row] : y_edges[row + 1],
                x_edges[col] : x_edges[col + 1],
            ]
            result[index] = patch.mean(axis=(1, 2))
            index += 1
    return result


@dataclass(frozen=True)
class PreprocessedSequences:
    stmap: np.ndarray
    motion_stmap: np.ndarray
    flow_stmap: np.ndarray
    motion_flow_stmap: np.ndarray


def build_sequences(
    face_frames_rgb: np.ndarray,
    rng: np.random.Generator,
    crop_config: SlidingCropConfig = SlidingCropConfig(),
) -> PreprocessedSequences:
    frames = _validate_frames(face_frames_rgb)
    motion_frames = sliding_crop_motion_augmentation(frames, rng, crop_config)

    sequences = PreprocessedSequences(
        stmap=build_stmap(frames),
        motion_stmap=build_stmap(motion_frames),
        flow_stmap=build_flow_stmap(dis_flow_magnitude(frames)),
        motion_flow_stmap=build_flow_stmap(dis_flow_magnitude(motion_frames)),
    )
    lengths = {
        sequences.stmap.shape[1],
        sequences.motion_stmap.shape[1],
        sequences.flow_stmap.shape[1],
        sequences.motion_flow_stmap.shape[1],
    }
    if lengths != {frames.shape[0]}:
        raise RuntimeError("Generated STMap sequences have different frame counts.")
    return sequences


def iter_windows(
    sequences: PreprocessedSequences,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
):
    frame_count = sequences.stmap.shape[1]
    shapes = (
        sequences.motion_stmap.shape[1],
        sequences.flow_stmap.shape[1],
        sequences.motion_flow_stmap.shape[1],
    )
    if any(length != frame_count for length in shapes):
        raise ValueError("All sequences must have the same frame count.")

    for start in range(0, frame_count - window_size + 1, stride):
        end = start + window_size
        yield {
            "start": start,
            "end": end,
            "stmap": scale_stmap_window(sequences.stmap[:, start:end, :]),
            "motion_stmap": scale_stmap_window(
                sequences.motion_stmap[:, start:end, :]
            ),
            "flow_stmap": sequences.flow_stmap[:, start:end].copy(),
            "motion_flow_stmap": sequences.motion_flow_stmap[:, start:end].copy(),
        }
