from dataclasses import dataclass
import numpy as np
import lap


@dataclass
class FaceTrack:
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    age: int = 0


def iou_batch(atlbrs, btlbrs):
    """
    Vectorized Pairwise Intersection over Union (IoU) computation.
    Guarantees non-mutating operations on input bounding box arrays.
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.empty((len(atlbrs), len(btlbrs)), dtype=np.float32)

    a = np.asarray(atlbrs, dtype=np.float32).copy()
    b = np.asarray(btlbrs, dtype=np.float32).copy()

    # Prevent invalid bounding boxes with negative width/height
    a[:, 2] = np.maximum(a[:, 2], a[:, 0])
    a[:, 3] = np.maximum(a[:, 3], a[:, 1])
    b[:, 2] = np.maximum(b[:, 2], b[:, 0])
    b[:, 3] = np.maximum(b[:, 3], b[:, 1])

    # Compute areas
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    # Broadcasted pairwise intersection boundaries
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def linear_assignment(cost_matrix, thresh):
    """
    Bipartite linear assignment engine using C++ Jonker-Volgenant algorithm (lap.lapjv).
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )

    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    matches = []
    unmatched_a = []
    unmatched_b = []

    for i, j in enumerate(x):
        if j >= 0:
            matches.append([i, j])
        else:
            unmatched_a.append(i)

    for j, i in enumerate(y):
        if i < 0:
            unmatched_b.append(j)

    return (
        np.array(matches, dtype=int),
        np.array(unmatched_a, dtype=int),
        np.array(unmatched_b, dtype=int),
    )


class LightweightFaceByteTracker:
    """
    Lightweight Face Identity Tracker designed for Deepfake Preprocessing:
    - 2-Stage High/Low Confidence Bipartite Matching
    - Fast C++ Jonker-Volgenant solver via lap.lapjv
    - Fully vectorized IoU matching with bounds checking
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_age: int = 30,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh  # Maximum cost distance threshold (1.0 - IoU)
        self.max_age = max_age
        self.tracks: dict[int, FaceTrack] = {}
        self.next_id = 0

    def _age_and_clean(self, matched_ids: set[int]):
        """Increments age for unmatched tracks and purges stale entries."""
        for t_id in list(self.tracks.keys()):
            if t_id not in matched_ids:
                self.tracks[t_id].age += 1
                if self.tracks[t_id].age > self.max_age:
                    del self.tracks[t_id]

    def update(self, detections) -> list[list]:
        """
        Input:
            detections: List or array of [[x1, y1, x2, y2, score], ...]
        Returns:
            active_tracks: List of [[x1, y1, x2, y2, score, track_id], ...] for current frame
        """
        # Defensive shape checking
        dets = np.asarray(detections, dtype=np.float32)
        if dets.ndim != 2 or dets.shape[1] < 5 or len(dets) == 0:
            self._age_and_clean(set())
            return []

        # Sanitize incoming bounding boxes
        dets[:, 0] = np.maximum(0, dets[:, 0])
        dets[:, 1] = np.maximum(0, dets[:, 1])
        dets[:, 2] = np.maximum(dets[:, 0], dets[:, 2])
        dets[:, 3] = np.maximum(dets[:, 1], dets[:, 3])

        scores = dets[:, 4]
        bboxes = dets[:, :4]

        # Stage Split: High vs Low Confidence Detections
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (scores < self.high_thresh)

        dets_high, scores_high = bboxes[high_mask], scores[high_mask]
        dets_low, scores_low = bboxes[low_mask], scores[low_mask]

        track_ids = list(self.tracks.keys())
        track_boxes = (
            np.array([self.tracks[t_id].bbox for t_id in track_ids])
            if track_ids
            else np.empty((0, 4))
        )

        matched_tracks = set()

        # -------------------------------------------------------------
        # STAGE 1: Match High-Confidence Detections against Active Tracks
        # -------------------------------------------------------------
        if len(track_boxes) > 0 and len(dets_high) > 0:
            cost_mat = 1.0 - iou_batch(track_boxes, dets_high)
            matches, u_tracks_idx, u_dets_high_idx = linear_assignment(
                cost_mat, thresh=self.match_thresh
            )

            for t_idx, d_idx in matches:
                t_id = track_ids[t_idx]
                self.tracks[t_id].bbox = dets_high[d_idx]
                self.tracks[t_id].score = float(scores_high[d_idx])
                self.tracks[t_id].age = 0
                matched_tracks.add(t_id)

            unmatched_track_ids = [track_ids[i] for i in u_tracks_idx]
        else:
            unmatched_track_ids = track_ids
            u_dets_high_idx = np.arange(len(dets_high))

        # -------------------------------------------------------------
        # STAGE 2: Recover Tracks using Low-Confidence Detections
        # -------------------------------------------------------------
        unmatched_track_boxes = (
            np.array([self.tracks[t_id].bbox for t_id in unmatched_track_ids])
            if unmatched_track_ids
            else np.empty((0, 4))
        )

        if len(unmatched_track_boxes) > 0 and len(dets_low) > 0:
            cost_mat_low = 1.0 - iou_batch(unmatched_track_boxes, dets_low)
            matches_low, _, _ = linear_assignment(
                cost_mat_low, thresh=self.match_thresh
            )

            for ut_idx, d_idx in matches_low:
                t_id = unmatched_track_ids[ut_idx]
                self.tracks[t_id].bbox = dets_low[d_idx]
                self.tracks[t_id].score = float(scores_low[d_idx])
                self.tracks[t_id].age = 0
                matched_tracks.add(t_id)

        # -------------------------------------------------------------
        # TRACK CREATION & AGE CLEANUP
        # -------------------------------------------------------------
        for d_idx in u_dets_high_idx:
            self.tracks[self.next_id] = FaceTrack(
                track_id=self.next_id,
                bbox=dets_high[d_idx],
                score=float(scores_high[d_idx]),
                age=0,
            )
            matched_tracks.add(self.next_id)
            self.next_id += 1

        self._age_and_clean(matched_tracks)

        # Return active tracks from current frame
        return [
            [*data.bbox, data.score, data.track_id]
            for data in self.tracks.values()
            if data.age == 0
        ]