from .club import Club

import os
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

class ClubAssigner:
    def __init__(self, club1: Club, club2: Club, images_to_save: int = 0, images_save_path: Optional[str] = None) -> None:
        """
        Initializes the ClubAssigner with club information and image saving parameters.

        Args:
            club1 (Club): The first club object.
            club2 (Club): The second club object.
            images_to_save (int): The number of images to save for analysis.
            images_save_path (Optional[str]): The directory path to save images.
        """
        self.club1 = club1
        self.club2 = club2
        self.model = ClubAssignerModel(self.club1, self.club2)
        self.club_colors: Dict[str, Any] = {
            club1.name: club1.player_jersey_color,
            club2.name: club2.player_jersey_color
        }
        self.goalkeeper_colors: Dict[str, Any] = {
            club1.name: club1.goalkeeper_jersey_color,
            club2.name: club2.goalkeeper_jersey_color
        }
        self.min_jersey_pixels = 30
        self.club_by_track: Dict[Tuple[str, int], str] = {}

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
            index, or (None, None) if the jersey color is not reliable in this frame.
        """
        color = self.get_jersey_color(frame, bbox, player_id, is_goalkeeper)
        if color is None:
            return None, None

        pred = self.model.predict(color, is_goalkeeper)
        
        return list(self.club_colors.keys())[pred], pred

    def assign_clubs(self, frame: np.ndarray, tracks: Dict[str, Dict[int, Any]]) -> Dict[str, Dict[int, Any]]:
        """
        Assign clubs to players and goalkeepers based on their jersey colors.

        Args:
            frame (np.ndarray): The current video frame.
            tracks (Dict[str, Dict[int, Any]]): The tracking data for players and goalkeepers.

        Returns:
            Dict[str, Dict[int, Any]]: The updated tracking data with assigned clubs.
        """
        tracks = tracks.copy()

        for track_type in ['goalkeeper', 'player']:
            for player_id, track in tracks[track_type].items():
                cache_key = (track_type, player_id)
                club = self.club_by_track.get(cache_key)
                if club is None:
                    bbox = track['bbox']
                    is_goalkeeper = (track_type == 'goalkeeper')
                    club, _ = self.get_player_club(frame, bbox, player_id, is_goalkeeper)
                    if club is None:
                        # Not enough reliable jersey pixels this frame; keep
                        # the track unassigned and retry on a later frame.
                        continue
                    self.club_by_track[cache_key] = club
                
                tracks[track_type][player_id]['club'] = club
                tracks[track_type][player_id]['club_color'] = self.club_colors[club]
        
        return tracks

class ClubAssignerModel:
    def __init__(self, club1: Club, club2: Club) -> None:
        """
        Initializes the ClubAssignerModel with jersey colors for the clubs.

        Args:
            club1 (Club): The first club object.
            club2 (Club): The second club object.
        """
        self.player_centroids = np.array([club1.player_jersey_color, club2.player_jersey_color])
        self.goalkeeper_centroids = np.array([club1.goalkeeper_jersey_color, club2.goalkeeper_jersey_color])

    def predict(self, extracted_color: Tuple[int, int, int], is_goalkeeper: bool = False) -> int:
        """
        Predict the club for a given jersey color based on the centroids.

        Args:
            extracted_color (Tuple[int, int, int]): The extracted jersey color in BGR format.
            is_goalkeeper (bool): Flag to indicate if the color is for a goalkeeper.

        Returns:
            int: The index of the predicted club (0 or 1).
        """
        if is_goalkeeper:
            centroids = self.goalkeeper_centroids
        else:
            centroids = self.player_centroids

        # Calculate distances
        distances = np.linalg.norm(extracted_color - centroids, axis=1)
        
        return np.argmin(distances)
