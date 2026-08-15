from .club import Club

import os
from collections import Counter, deque
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List

class ClubAssigner:
    def __init__(self, club1: Club, club2: Club, images_to_save: int = 0, images_save_path: Optional[str] = None,
                 referee_assign_dist: float = 85.0, referee_color: Optional[Tuple[int, int, int]] = None,
                 referee_hue_tol_deg: float = 15.0) -> None:
        """
        Initializes the ClubAssigner with club information and image saving parameters.

        Args:
            club1 (Club): The first club object.
            club2 (Club): The second club object.
            images_to_save (int): The number of images to save for analysis.
            images_save_path (Optional[str]): The directory path to save images.
            referee_assign_dist (float): Maximum weighted HSV distance to a club
                reference for a sample to be assigned to that club; farther
                samples are referee candidates.
            referee_color (Optional[Tuple[int, int, int]]): RGB reference color
                of the referee jersey. When given, a sample whose hue is within
                `referee_hue_tol_deg` of the referee reference hue (and closer
                than both club hues) is a referee.
            referee_hue_tol_deg (float): Maximum circular hue distance to the
                referee reference hue for a referee assignment.
        """
        self.club1 = club1
        self.club2 = club2
        self.model = ClubAssignerModel(self.club1, self.club2, referee_assign_dist, referee_color, referee_hue_tol_deg)
        self.club_colors: Dict[str, Any] = {
            club1.name: club1.player_jersey_color,
            club2.name: club2.player_jersey_color
        }
        self.goalkeeper_colors: Dict[str, Any] = {
            club1.name: club1.goalkeeper_jersey_color,
            club2.name: club2.goalkeeper_jersey_color
        }
        self.min_jersey_pixels = 30
        self.frame_cluster_min_samples = 4
        self.cluster_separation_min_deg = 25.0
        self.vote_window = 60
        self.referee_assign_dist = float(referee_assign_dist)
        self.referee_color = referee_color
        self.referee_hue_tol_deg = float(referee_hue_tol_deg)
        self.club_by_track: Dict[Tuple[str, int], str] = {}
        self.votes_by_track: Dict[Tuple[str, int], deque] = {}
        self.player_ref_hues: Dict[str, float] = {
            name: self._rgb_to_hue(color) for name, color in self.club_colors.items()
        }

        # Saving images for analysis
        self.images_to_save = images_to_save
        self.output_dir = images_save_path

        if not images_save_path:
            images_to_save = 0
            self.saved_images = 0
        else:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        
            self.saved_images = len([name for name in os.listdir(self.output_dir) if name.startswith('player')])

    @staticmethod
    def _rgb_to_hue(color: Tuple[int, int, int]) -> float:
        """Convert an RGB color to OpenCV hue (0-180)."""
        bgr = np.uint8([[[color[2], color[1], color[0]]]])
        return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 0])

    @staticmethod
    def _hue_dist(a: float, b: float) -> float:
        """Circular hue distance in degrees (OpenCV hue range 0-180)."""
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    @staticmethod
    def _stats_to_rgb(hue: float, sat: float, val: float) -> Tuple[int, int, int]:
        """Convert median HSV statistics back to an RGB color."""
        bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    @staticmethod
    def _circular_2means(
        hues: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministic 2-means on circular hue data (sin/cos embedding),
        initialized with the farthest sample pair and refined with Lloyd
        iterations. Handles the red hue wrap-around (0/180) naturally.

        Args:
            hues: Array of OpenCV hue values (0-180).
            weights: Sample weights (e.g., valid pixel counts).

        Returns:
            Tuple of (labels array, 2x2 unit-vector cluster centers).
        """
        n = len(hues)
        if n < 2:
            return np.zeros(n, dtype=int), np.zeros((2, 2))

        ang = np.deg2rad(hues * 2.0)
        x = np.stack([np.cos(ang), np.sin(ang)], axis=1)

        # Farthest-pair initialization (deterministic)
        best = (0, 1, -1.0)
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(x[i] - x[j]))
                if dist > best[2]:
                    best = (i, j, dist)
        centers = np.stack([x[best[0]], x[best[1]]])

        labels = np.zeros(n, dtype=int)
        for _ in range(10):
            new_labels = np.argmin(
                np.stack(
                    [np.linalg.norm(x - centers[0], axis=1),
                     np.linalg.norm(x - centers[1], axis=1)],
                    axis=1,
                ),
                axis=1,
            )
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(2):
                mask = labels == k
                if mask.any():
                    weighted = (x[mask] * weights[mask][:, None]).sum(axis=0)
                    norm = np.linalg.norm(weighted)
                    if norm > 1e-9:
                        centers[k] = weighted / norm

        return labels, centers

    def _assign_frame(
        self,
        items: List[Tuple[Tuple[str, int], float, float, float, int]],
    ) -> Dict[Tuple[str, int], str]:
        """
        Assign every sample of the current frame to a club name or 'referee'.

        Referee candidates are samples whose weighted HSV distance to BOTH club
        references exceeds `referee_assign_dist`; they vote 'referee' directly
        and the per-track sliding window requires a consistent majority before
        a player is flagged as a referee, absorbing single-frame noise. The
        remaining club samples are clustered into two hue groups and mapped to
        clubs via the configured reference jersey colors.

        Args:
            items: List of (cache_key, median_hue, median_sat, median_val, n_pixels).

        Returns:
            Dict mapping cache_key to a club name or 'referee'. Abstained keys
            are absent.
        """
        result: Dict[Tuple[str, int], str] = {}

        referee_items: List[Tuple[Tuple[str, int], float, float, float, int]] = []
        club_items: List[Tuple[Tuple[str, int], float, float, float, int]] = []
        for it in items:
            color = self._stats_to_rgb(it[1], it[2], it[3])
            if self.model.predict_referee(color, is_goalkeeper=False) is None:
                referee_items.append(it)
            else:
                club_items.append(it)

        for it in referee_items:
            result[it[0]] = "referee"

        if len(club_items) >= self.frame_cluster_min_samples:
            hues = np.array([it[1] for it in club_items], dtype=float)
            weights = np.array([it[4] for it in club_items], dtype=float)
            labels, centers = self._circular_2means(hues, weights)

            center_hues = (np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0])) % 360.0) / 2.0
            refs = list(self.player_ref_hues.items())

            separation = self._hue_dist(center_hues[0], center_hues[1])
            if separation >= self.cluster_separation_min_deg:
                nearest = [
                    min(range(2), key=lambda r: self._hue_dist(center_hues[i], refs[r][1]))
                    for i in range(2)
                ]
                if nearest[0] != nearest[1]:
                    mapping = {i: refs[nearest[i]][0] for i in range(2)}
                    for it, label in zip(club_items, labels):
                        result[it[0]] = mapping[int(label)]
                    return result

        # Clusters not separable (single team in frame) or mapping ambiguous:
        # fall back to per-sample nearest-reference classification with a
        # referee rejection threshold.
        for it in club_items:
            color = self._stats_to_rgb(it[1], it[2], it[3])
            pred = self.model.predict_referee(color, is_goalkeeper=False)
            if pred is not None:
                result[it[0]] = list(self.club_colors.keys())[pred]
        return result

    def apply_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Remove pitch-green background pixels from a player crop.

        Args:
            image (np.ndarray): An image to apply the mask to.

        Returns:
            np.ndarray: The masked image (green pixels set to black).
        """
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the green color range in HSV
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])

        # Create the mask and keep only non-green pixels
        mask = cv2.bitwise_not(cv2.inRange(hsv_img, lower_green, upper_green))
        return cv2.bitwise_and(image, image, mask=mask)

    def extract_jersey_stats(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Tuple[float, float, float, int]]:
        """
        Extract robust jersey color statistics from a player's torso band.

        Samples the central torso band of the bounding box (avoids head,
        legs and bbox-edge background), removes green and shadowed pixels,
        and returns the median hue/saturation/value plus the number of
        valid pixels.

        Args:
            frame (np.ndarray): The current video frame.
            bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).

        Returns:
            Optional[Tuple[float, float, float, int]]: (median_hue, median_sat, median_val, n_pixels)
            or None if the crop is empty.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox)
        img = frame[y1:y2, x1:x2]
        if img.size == 0:
            return None

        h, w = img.shape[:2]
        # Torso band: horizontal 20%-80%, vertical 30%-60% of the bbox
        band = img[int(h * 0.30):int(h * 0.60), int(w * 0.20):int(w * 0.80)]
        if band.size == 0:
            return None

        masked = self.apply_mask(band)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Keep only colored, non-shadowed jersey pixels
        keep = (sat >= 60) & (val >= 50)
        if not keep.any():
            return (0.0, 0.0, 0.0, 0)

        return (
            float(np.median(hue[keep])),
            float(np.median(sat[keep])),
            float(np.median(val[keep])),
            int(keep.sum()),
        )

    def extract_dark_jersey_stats(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Tuple[float, float, float, int]]:
        """
        Extract stats for dark jerseys (e.g. a black goalkeeper) from the
        torso band's low-brightness, low-saturation pixels. Returns None when
        the band has no sufficiently large dark region.

        Args:
            frame (np.ndarray): The current video frame.
            bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).

        Returns:
            Optional[Tuple[float, float, float, int]]: (median_hue, median_sat,
            median_val, n_pixels) or None when no dark jersey region exists.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox)
        img = frame[y1:y2, x1:x2]
        if img.size == 0:
            return None

        h, w = img.shape[:2]
        band = img[int(h * 0.30):int(h * 0.60), int(w * 0.20):int(w * 0.80)]
        if band.size == 0:
            return None

        masked = self.apply_mask(band)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Dark jersey pixels: low brightness, not strongly saturated
        keep = (val >= 25) & (val < 105) & (sat < 90)
        if not keep.any():
            return (0.0, 0.0, 0.0, 0)

        return (
            float(np.median(hue[keep])),
            float(np.median(sat[keep])),
            float(np.median(val[keep])),
            int(keep.sum()),
        )

    def _assign_dark_sample(
        self, stats: Tuple[float, float, float, int]
    ) -> Optional[str]:
        """
        Classify a dark-jersey sample: the nearest of the four references
        (two club, two goalkeeper) decides. Goalkeeper references win with a
        wider threshold (60) because their colors are the dark jerseys;
        club references need a tight threshold (45) to avoid shadowed field
        players drifting onto the wrong team.

        Returns:
            Optional[str]: A club name, or None to abstain this frame.
        """
        color = self._stats_to_rgb(stats[0], stats[1], stats[2])
        hsv = self.model._rgb_to_hsv(color)
        d_gk = [self.model._hsv_distance(hsv, r) for r in self.model.goalkeeper_hsv]
        d_club = [self.model._hsv_distance(hsv, r) for r in self.model.player_hsv]
        gk_best = int(np.argmin(d_gk))
        club_best = int(np.argmin(d_club))
        names = list(self.club_colors.keys())

        if self.model._is_referee_color(hsv, min_sat=30.0, min_val=25.0):
            return "referee"

        if (
            d_gk[gk_best] <= self.model._gk_threshold(self.model.goalkeeper_hsv[gk_best])
            and d_gk[gk_best] < d_club[club_best]
        ):
            return names[gk_best]
        if d_club[club_best] <= 45.0:
            return names[club_best]
        return None

    def save_player_image(self, img: np.ndarray, player_id: int, is_goalkeeper: bool = False) -> None:
        """
        Save the player's image to the specified directory.

        Args:
            img (np.ndarray): The image of the player.
            player_id (int): The unique identifier for the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.
        """
        # Use 'goalkeeper' or 'player' prefix based on is_goalkeeper flag
        prefix = 'goalkeeper' if is_goalkeeper else 'player'
        filename = os.path.join(self.output_dir, f"{prefix}_{player_id}.png")
        if os.path.exists(filename):
            return
        cv2.imwrite(filename, img)
        print(f"Saved {prefix} image: {filename}")
        # Increment the count of saved images
        self.saved_images += 1

    def get_jersey_color(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], player_id: int, is_goalkeeper: bool = False) -> Optional[Tuple[int, int, int]]:
        """
        Extract the jersey color from a player's bounding box in the frame.

        Args:
            frame (np.ndarray): The current video frame.
            bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
            player_id (int): The unique identifier for the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.

        Returns:
            Optional[Tuple[int, int, int]]: The jersey color in RGB format, or None if
            there are not enough reliable jersey pixels in this frame.
        """
        # Save player images only if needed
        if self.saved_images < self.images_to_save:
            img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            img_top = img[0:img.shape[0] // 2, :] 
            self.save_player_image(img_top, player_id, is_goalkeeper)  # Pass is_goalkeeper here

        stats = self.extract_jersey_stats(frame, bbox)
        if stats is None:
            return None

        median_hue, median_sat, median_val, n_pixels = stats
        if n_pixels < self.min_jersey_pixels:
            return None

        hsv = np.uint8([[[median_hue, median_sat, median_val]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    def get_player_club(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], player_id: int, is_goalkeeper: bool = False) -> Tuple[Optional[str], Optional[int]]:
        """
        Determine the club associated with a player based on their jersey color.

        Args:
            frame (np.ndarray): The current video frame.
            bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
            player_id (int): The unique identifier for the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.

        Returns:
            Tuple[Optional[str], Optional[int]]: The club name and the predicted class
            index, or (None, None) if the jersey color is not reliable in this frame
            or is a referee color.
        """
        color = self.get_jersey_color(frame, bbox, player_id, is_goalkeeper)
        if color is None:
            return None, None

        pred = self.model.predict_referee(color, is_goalkeeper)
        if pred is None:
            return None, None

        return list(self.club_colors.keys())[pred], pred

    def assign_clubs(self, frame: np.ndarray, tracks: Dict[str, Dict[int, Any]]) -> Dict[str, Dict[int, Any]]:
        """
        Assign clubs to players, goalkeepers, and referee-class tracks based on
        their jersey colors.

        Referee-class tracks are classified with the same color logic as
        players: when their jersey matches a club reference they are restored
        to that club (drawn as a club player); when they stay far from both
        club colors they keep the referee identity. Player-class tracks whose
        color is far from both clubs (e.g. a yellow referee misdetected as a
        player) are flagged as referees and lose their club color.

        Args:
            frame (np.ndarray): The current video frame.
            tracks (Dict[str, Dict[int, Any]]): The tracking data for players and goalkeepers.

        Returns:
            Dict[str, Dict[int, Any]]: The updated tracking data with assigned clubs.
        """
        tracks = tracks.copy()

        samples: List[Tuple[Tuple[str, int], str, Tuple[float, float, float, int], bool]] = []
        for track_type in ['goalkeeper', 'player', 'referee']:
            tracks.setdefault(track_type, {})
            for player_id, track in tracks[track_type].items():
                cache_key = (track_type, player_id)
                stats = self.extract_jersey_stats(frame, track['bbox'])
                is_dark = False
                if stats is None or stats[3] < self.min_jersey_pixels:
                    # Dark jersey (e.g. a black goalkeeper): bright pixels are
                    # absent, fall back to the low-brightness band.
                    stats = self.extract_dark_jersey_stats(frame, track['bbox'])
                    is_dark = stats is not None and stats[3] >= self.min_jersey_pixels
                if stats is None or stats[3] < self.min_jersey_pixels:
                    # Not enough reliable jersey pixels this frame; keep the
                    # previous majority as the displayed club.
                    continue
                samples.append((cache_key, track_type, stats, is_dark))

        if samples:
            player_items = [s for s in samples if s[1] != 'goalkeeper']
            goalkeeper_items = [s for s in samples if s[1] == 'goalkeeper']

            frame_decisions: Dict[Tuple[str, int], str] = {}
            bright_items = [s for s in player_items if not s[3]]
            dark_items = [s for s in player_items if s[3]]

            for key, _, stats, _ in dark_items:
                decision = self._assign_dark_sample(stats)
                if decision is not None:
                    frame_decisions[key] = decision
                # Otherwise abstain this frame; the window keeps the majority.

            if len(bright_items) >= self.frame_cluster_min_samples:
                frame_items = [
                    (key, stats[0], stats[1], stats[2], stats[3])
                    for key, _, stats, _ in bright_items
                ]
                frame_decisions.update(self._assign_frame(frame_items))
            else:
                for key, _, stats, _ in bright_items:
                    color = self._stats_to_rgb(stats[0], stats[1], stats[2])
                    pred = self.model.predict_referee(color, is_goalkeeper=False)
                    if pred is None:
                        frame_decisions[key] = "referee"
                    else:
                        frame_decisions[key] = list(self.club_colors.keys())[pred]

            for key, _, stats, is_dark in goalkeeper_items:
                color = self._stats_to_rgb(stats[0], stats[1], stats[2])
                if is_dark:
                    pred = self._assign_dark_sample(stats)
                    if pred is not None:
                        frame_decisions[key] = pred
                else:
                    pred = self.model.predict(color, is_goalkeeper=True)
                    frame_decisions[key] = list(self.club_colors.keys())[pred]

            for cache_key, club in frame_decisions.items():
                window = self.votes_by_track.setdefault(
                    cache_key, deque(maxlen=self.vote_window)
                )
                window.append(club)
                # Running majority over the recent window; tracks identity
                # switches or drifting colors with bounded latency.
                self.club_by_track[cache_key] = Counter(window).most_common(1)[0][0]

        # Write the current majority club (or keep previous) for every track.
        for track_type in ['goalkeeper', 'player', 'referee']:
            for player_id, track in tracks.setdefault(track_type, {}).items():
                cache_key = (track_type, player_id)
                vote = self.club_by_track.get(cache_key)
                if vote is None:
                    continue
                if vote == 'referee':
                    track['referee'] = True
                    track.pop('club', None)
                    track.pop('club_color', None)
                else:
                    track['club'] = vote
                    track['club_color'] = self.club_colors[vote]
                    track.pop('referee', None)

        return tracks

class ClubAssignerModel:
    def __init__(self, club1: Club, club2: Club, referee_assign_dist: float = 85.0,
                 referee_color: Optional[Tuple[int, int, int]] = None,
                 referee_hue_tol_deg: float = 15.0,
                 gk_match_dist: float = 60.0,
                 gk_dark_match_dist: float = 40.0) -> None:
        """
        Initializes the ClubAssignerModel with jersey colors for the clubs.

        Args:
            club1 (Club): The first club object.
            club2 (Club): The second club object.
            referee_assign_dist (float): Maximum weighted HSV distance to a club
                reference for a club assignment; farther colors are referees.
            referee_color (Optional[Tuple[int, int, int]]): RGB reference color
                of the referee jersey.
            referee_hue_tol_deg (float): Maximum circular hue distance to the
                referee reference hue for a referee assignment (brightness and
                saturation of the sample must also be reasonable).
            gk_match_dist (float): Maximum weighted HSV distance to a colored
                goalkeeper reference for a player-class sample to be that
                team's goalkeeper.
            gk_dark_match_dist (float): Same for dark goalkeeper references
                (lower brightness); dark references match only dark samples.
        """
        self.player_centroids = np.array([club1.player_jersey_color, club2.player_jersey_color])
        self.goalkeeper_centroids = np.array([club1.goalkeeper_jersey_color, club2.goalkeeper_jersey_color])
        self.player_hsv = np.array([self._rgb_to_hsv(c) for c in self.player_centroids])
        self.goalkeeper_hsv = np.array([self._rgb_to_hsv(c) for c in self.goalkeeper_centroids])
        self.referee_assign_dist = float(referee_assign_dist)
        self.referee_hue_tol_deg = float(referee_hue_tol_deg)
        self.gk_match_dist = float(gk_match_dist)
        self.gk_dark_match_dist = float(gk_dark_match_dist)
        self.referee_hsv = (
            np.array(self._rgb_to_hsv(referee_color)) if referee_color is not None else None
        )

    def _is_referee_color(self, hsv, min_sat: float = 60.0, min_val: float = 60.0) -> bool:
        """Hue-based referee color check, robust to lighting.

        A sample is a referee color only when its hue is within
        `referee_hue_tol_deg` of the referee reference hue, is reasonably
        saturated/bright, and is closer to the referee hue than to both club
        hues. Weighted distance is deliberately not used here: bright colors
        near the hue circle (e.g. a bright red jersey) would otherwise be
        pulled to the referee reference by the brightness term.
        """
        if self.referee_hsv is None:
            return False
        if float(hsv[1]) < min_sat or float(hsv[2]) < min_val:
            return False
        dh_ref = ClubAssigner._hue_dist(float(hsv[0]), float(self.referee_hsv[0]))
        if dh_ref > self.referee_hue_tol_deg:
            return False
        dh_clubs = [ClubAssigner._hue_dist(float(hsv[0]), float(r[0])) for r in self.player_hsv]
        return dh_ref < min(dh_clubs)

    def _gk_threshold(self, ref: np.ndarray) -> float:
        """Match threshold for a goalkeeper reference (dark refs match tighter)."""
        return self.gk_match_dist if float(ref[2]) >= 60.0 else self.gk_dark_match_dist

    @staticmethod
    def _rgb_to_hsv(color: Tuple[int, int, int]) -> Tuple[float, float, float]:
        bgr = np.uint8([[[color[2], color[1], color[0]]]])
        h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
        return (float(h), float(s), float(v))

    @staticmethod
    def _hsv_distance(
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        """
        Weighted HSV distance: circular hue difference dominates when both
        colors are saturated; saturation/value differences add robustness
        for near-gray colors where hue is meaningless.
        """
        hue_weight = 2.0 if (a[1] >= 30.0 and b[1] >= 30.0) else 0.0
        dh = ClubAssigner._hue_dist(a[0], b[0])
        return hue_weight * dh + 0.25 * abs(a[1] - b[1]) + 0.25 * abs(a[2] - b[2])

    def predict(self, extracted_color: Tuple[int, int, int], is_goalkeeper: bool = False) -> int:
        """
        Predict the club for a given jersey color based on the centroids.

        Args:
            extracted_color (Tuple[int, int, int]): The extracted jersey color in RGB format.
            is_goalkeeper (bool): Flag to indicate if the color is for a goalkeeper.

        Returns:
            int: The index of the predicted club (0 or 1).
        """
        if is_goalkeeper:
            reference = self.goalkeeper_hsv
        else:
            reference = self.player_hsv

        hsv = self._rgb_to_hsv(extracted_color)
        distances = np.array([self._hsv_distance(hsv, r) for r in reference])
        
        return int(np.argmin(distances))

    def predict_referee(
        self, extracted_color: Tuple[int, int, int], is_goalkeeper: bool = False
    ) -> Optional[int]:
        """
        Predict the club for a jersey color, or return None when the color is
        a referee color rather than a club color.

        Player path: a sample matching a goalkeeper reference (within
        `gk_match_dist` and closer than both club references) is that team's
        goalkeeper. With a configured referee reference color, a sample is a
        referee when its distance to the referee reference is smaller than its
        distance to the club references and within `referee_match_dist`.
        Without a referee reference, a sample is a referee when it is farther
        than `referee_assign_dist` from both club references.

        Args:
            extracted_color (Tuple[int, int, int]): The extracted jersey color in RGB format.
            is_goalkeeper (bool): Flag to indicate if the color is for a goalkeeper.

        Returns:
            Optional[int]: The predicted club index (0 or 1), or None for a referee.
        """
        hsv = self._rgb_to_hsv(extracted_color)

        if is_goalkeeper:
            gk_distances = np.array([self._hsv_distance(hsv, r) for r in self.goalkeeper_hsv])
            best = int(np.argmin(gk_distances))
            if self._is_referee_color(hsv, min_sat=30.0, min_val=25.0):
                return None
            return best

        club_distances = np.array([self._hsv_distance(hsv, r) for r in self.player_hsv])
        best = int(np.argmin(club_distances))

        # Referee reference check first: hue-based, so lighting shifts on the
        # referee jersey do not lose the match, while other bright colors
        # (e.g. a red substitute bib) are not stolen by the brightness term.
        if self._is_referee_color(hsv):
            return None

        # Goalkeeper reference match: a player-class sample whose jersey
        # matches a goalkeeper reference is that team's goalkeeper.
        gk_distances = np.array([self._hsv_distance(hsv, r) for r in self.goalkeeper_hsv])
        gk_best = int(np.argmin(gk_distances))
        if (
            float(gk_distances[gk_best]) <= self._gk_threshold(self.goalkeeper_hsv[gk_best])
            and float(gk_distances[gk_best]) < float(club_distances[best])
        ):
            return gk_best

        if float(club_distances[best]) > self.referee_assign_dist:
            return None
        return best
