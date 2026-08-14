"""Synthetic tests for team color classification (Phase 1-3).

Covers: torso-band hue extraction with shadow/grass contamination, frame-level
circular 2-means grouping, sliding-window temporal voting, and the single-team
frame fallback.
"""
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from club_assignment import Club, ClubAssigner, ClubAssignerModel  # noqa: E402

MAROON_RGB = (120, 37, 66)
NAVY_RGB = (31, 72, 127)
YELLOW_RGB = (255, 220, 30)


def make_frame(players, size=(1280, 720)):
    """Render a synthetic pitch frame with player jerseys.

    players: list of (bbox, rgb) where bbox=(x1,y1,x2,y2).
    Background is green grass with mild noise; jerseys are pure color with
    added noise and a dark shadow band to simulate lighting.
    """
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # grass background (BGR: green channel dominant)
    rng = np.random.default_rng(7)
    frame[:, :, 1] = rng.integers(90, 140, size=(size[1], size[0]))
    frame[:, :, 0] = rng.integers(30, 60, size=(size[1], size[0]))
    frame[:, :, 2] = rng.integers(20, 50, size=(size[1], size[0]))

    rng = np.random.default_rng(11)
    for (x1, y1, x2, y2), rgb in players:
        h, w = y2 - y1, x2 - x1
        # skin-colored head
        frame[y1:y1 + int(h * 0.3), x1:x2] = (120, 150, 190)
        # jersey torso (30%-60% vertical)
        jersey = np.zeros((int(h * 0.3), w, 3), dtype=np.uint8)
        jersey[:] = rgb[::-1]
        noise = rng.integers(-25, 25, size=jersey.shape)
        jersey = np.clip(jersey.astype(int) + noise, 0, 255).astype(np.uint8)
        frame[y1 + int(h * 0.3):y1 + int(h * 0.6), x1:x2] = jersey
        # dark shadow band on one edge (simulates uneven lighting)
        frame[y1 + int(h * 0.3):y1 + int(h * 0.6), x2 - max(2, w // 5):x2] = (
            frame[y1 + int(h * 0.3):y1 + int(h * 0.6), x2 - max(2, w // 5):x2] // 3
        )
        # legs
        frame[y1 + int(h * 0.6):y2, x1:x2] = (40, 40, 40)
    return frame


def make_players(team_rgbs, n_per_team=5, seed=3):
    """Layout n_per_team players of each team across the frame."""
    rng = np.random.default_rng(seed)
    players = []
    xs = np.linspace(100, 1100, n_per_team * 2)
    for i, rgb in enumerate(team_rgbs):
        for j in range(n_per_team):
            x1 = int(xs[i * n_per_team + j])
            y1 = int(300 + rng.integers(-30, 30))
            players.append(((x1, y1, x1 + 55, y1 + 120), rgb))
    return players


def test_torso_extraction_recovers_jersey_color():
    assigner = ClubAssigner(Club("Maroon", MAROON_RGB, (80, 80, 80)),
                            Club("Navy", NAVY_RGB, (80, 80, 80)))
    players = make_players([MAROON_RGB, NAVY_RGB])
    frame = make_frame(players)
    for (bbox, rgb) in players:
        stats = assigner.extract_jersey_stats(frame, bbox)
        assert stats is not None and stats[3] >= 30, "jersey pixels must be found"
        hue = stats[0]
        ref_hue = assigner._rgb_to_hue(rgb)
        assert assigner._hue_dist(hue, ref_hue) < 20, (
            f"extracted hue {hue:.0f} too far from jersey hue {ref_hue:.0f}"
        )


def test_frame_cluster_groups_two_teams():
    assigner = ClubAssigner(Club("Maroon", MAROON_RGB, (80, 80, 80)),
                            Club("Navy", NAVY_RGB, (80, 80, 80)))
    players = make_players([MAROON_RGB, NAVY_RGB])
    frame = make_frame(players)
    tracks = {
        "player": {
            i: {"bbox": list(bbox)} for i, (bbox, _rgb) in enumerate(players)
        },
        "goalkeeper": {},
    }
    out = assigner.assign_clubs(frame, tracks)
    for i, (bbox, rgb) in enumerate(players):
        expected = "Maroon" if rgb == MAROON_RGB else "Navy"
        assert out["player"][i]["club"] == expected, f"player {i} misassigned"


def test_single_team_frame_falls_back_to_nearest_reference():
    assigner = ClubAssigner(Club("Maroon", MAROON_RGB, (80, 80, 80)),
                            Club("Navy", NAVY_RGB, (80, 80, 80)))
    players = make_players([MAROON_RGB, MAROON_RGB])  # only maroon on frame
    frame = make_frame(players)
    tracks = {
        "player": {i: {"bbox": list(bbox)} for i, (bbox, _rgb) in enumerate(players)},
        "goalkeeper": {},
    }
    out = assigner.assign_clubs(frame, tracks)
    for i in tracks["player"]:
        assert out["player"][i]["club"] == "Maroon", f"player {i} should be Maroon"


def test_sliding_window_recovers_from_id_switch():
    assigner = ClubAssigner(Club("Maroon", MAROON_RGB, (80, 80, 80)),
                            Club("Navy", NAVY_RGB, (80, 80, 80)))
    # player 0 wears maroon for 40 frames, then navy for 80 frames
    maroon_frames = [make_frame([((500, 300, 555, 420), MAROON_RGB)])
                     for _ in range(40)]
    navy_frames = [make_frame([((500, 300, 555, 420), NAVY_RGB)])
                   for _ in range(80)]
    tracks = {"player": {0: {"bbox": [500, 300, 555, 420]}}, "goalkeeper": {}}
    for frame in maroon_frames:
        out = assigner.assign_clubs(frame, tracks)
    assert out["player"][0]["club"] == "Maroon"
    for frame in navy_frames:
        out = assigner.assign_clubs(frame, tracks)
    assert out["player"][0]["club"] == "Navy", (
        "sliding window must absorb the identity switch"
    )


def test_circular_2means_wraps_around_red():
    assigner = ClubAssigner(Club("Maroon", MAROON_RGB, (80, 80, 80)),
                            Club("Navy", NAVY_RGB, (80, 80, 80)))
    hues = np.array([2.0, 3.0, 4.0, 177.0, 178.0, 179.0])
    weights = np.ones(6)
    labels, centers = assigner._circular_2means(hues, weights)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_low_confidence_crop_yields_none():
    assigner = ClubAssigner(Club("Maroon", MAROON_RGB, (80, 80, 80)),
                            Club("Navy", NAVY_RGB, (80, 80, 80)))
    frame = make_frame([])  # empty pitch
    bbox = (600, 350, 615, 365)  # tiny far-away bbox, mostly grass
    assert assigner.get_jersey_color(frame, bbox, 0) is None


def _make_assigner():
    return ClubAssigner(Club("Maroon", MAROON_RGB, (80, 80, 80)),
                        Club("Navy", NAVY_RGB, (80, 80, 80)))


def test_yellow_player_flagged_as_referee():
    assigner = _make_assigner()
    players = make_players([MAROON_RGB, NAVY_RGB])
    players.append(((600, 300, 655, 420), YELLOW_RGB))
    frame = make_frame(players)
    tracks = {
        "player": {i: {"bbox": list(bbox)} for i, (bbox, _rgb) in enumerate(players)},
        "goalkeeper": {},
        "referee": {},
    }
    out = None
    for _ in range(35):
        out = assigner.assign_clubs(frame, tracks)
    yellow_id = len(players) - 1
    assert out["player"][yellow_id].get("referee") is True
    assert out["player"][yellow_id].get("club") is None
    for i in range(len(players) - 1):
        expected = "Maroon" if i < 5 else "Navy"
        assert out["player"][i]["club"] == expected, f"player {i} misassigned"


def test_referee_track_with_club_color_restored():
    assigner = _make_assigner()
    players = make_players([MAROON_RGB, NAVY_RGB])
    players.append(((600, 300, 655, 420), NAVY_RGB))  # navy player in referee class
    frame = make_frame(players)
    tracks = {
        "player": {i: {"bbox": list(bbox)} for i, (bbox, _rgb) in enumerate(players[:-1])},
        "goalkeeper": {},
        "referee": {0: {"bbox": list(players[-1][0])}},
    }
    out = None
    for _ in range(35):
        out = assigner.assign_clubs(frame, tracks)
    assert out["referee"][0]["club"] == "Navy"


def test_yellow_referee_track_stays_referee():
    assigner = _make_assigner()
    players = make_players([MAROON_RGB, NAVY_RGB])
    players.append(((600, 300, 655, 420), YELLOW_RGB))  # real referee
    frame = make_frame(players)
    tracks = {
        "player": {i: {"bbox": list(bbox)} for i, (bbox, _rgb) in enumerate(players[:-1])},
        "goalkeeper": {},
        "referee": {0: {"bbox": list(players[-1][0])}},
    }
    out = None
    for _ in range(35):
        out = assigner.assign_clubs(frame, tracks)
    assert out["referee"][0].get("club") is None
    assert out["referee"][0].get("referee", True) is not False


def test_club_players_never_flagged_referee():
    assigner = _make_assigner()
    players = make_players([MAROON_RGB, NAVY_RGB])
    tracks = {
        "player": {i: {"bbox": list(bbox)} for i, (bbox, _rgb) in enumerate(players)},
        "goalkeeper": {},
        "referee": {},
    }
    for _ in range(35):
        frame = make_frame(players)
        out = assigner.assign_clubs(frame, tracks)
        for i in tracks["player"]:
            assert not out["player"][i].get("referee"), f"player {i} wrongly flagged"


def test_referee_flag_recovers_after_jersey_change():
    assigner = _make_assigner()
    yellow_frames = [make_frame([((500, 300, 555, 420), YELLOW_RGB)])
                     for _ in range(40)]
    navy_frames = [make_frame([((500, 300, 555, 420), NAVY_RGB)])
                   for _ in range(80)]
    tracks = {
        "player": {0: {"bbox": [500, 300, 555, 420]}},
        "goalkeeper": {},
        "referee": {},
    }
    for frame in yellow_frames:
        out = assigner.assign_clubs(frame, tracks)
    assert out["player"][0].get("referee") is True
    for frame in navy_frames:
        out = assigner.assign_clubs(frame, tracks)
    assert out["player"][0].get("club") == "Navy"
    assert not out["player"][0].get("referee"), "referee flag must clear after jersey change"


def _hsv_color(hue, sat, val):
    bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def test_referee_assign_dist_boundary_zone():
    """Samples in the calibrated gap (83-88) must land on the right side.

    Measured on demo2: maroon players reach min-distance 83, yellow referees
    start at 88; the threshold (85) must reject the former side and accept the
    latter.
    """
    assigner = _make_assigner()
    model = assigner.model
    club_seen = referee_seen = False
    ref0 = model.player_hsv[0]  # maroon reference
    for hue in range(135, 172):
        rgb = _hsv_color(float(hue), 80.0, 60.0)
        hsv = model._rgb_to_hsv(rgb)
        dmin = min(model._hsv_distance(hsv, r) for r in model.player_hsv)
        pred = model.predict_referee(rgb, is_goalkeeper=False)
        if 78.0 <= dmin <= 84.0:
            assert pred is not None, f"dmin {dmin:.0f} must stay a club color"
            club_seen = True
        if 86.0 <= dmin <= 95.0:
            assert pred is None, f"dmin {dmin:.0f} must be a referee color"
            referee_seen = True
    assert club_seen, "no sample landed in the club boundary zone"
    assert referee_seen, "no sample landed in the referee boundary zone"


def test_referee_color_reference_wins_when_closer():
    # Explicit three-reference configuration (demo2 measured referee jersey)
    model = ClubAssignerModel(
        Club('Maroon', MAROON_RGB, (80, 80, 80)),
        Club('Navy', NAVY_RGB, (80, 80, 80)),
        referee_assign_dist=85.0,
        referee_color=(168, 156, 74),
    )
    assert model.predict_referee((168, 156, 74), is_goalkeeper=False) is None
    assert model.predict_referee((170, 158, 70), is_goalkeeper=False) is None
    assert model.predict_referee(MAROON_RGB, is_goalkeeper=False) == 0
    assert model.predict_referee(NAVY_RGB, is_goalkeeper=False) == 1
    assert model.predict_referee(YELLOW_RGB, is_goalkeeper=False) is None


def test_get_player_club_rejects_referee_color():
    assigner = _make_assigner()
    players = make_players([MAROON_RGB, NAVY_RGB])
    players.append(((600, 300, 655, 420), YELLOW_RGB))
    frame = make_frame(players)
    bbox = players[-1][0]
    club, pred = assigner.get_player_club(frame, bbox, 99)
    assert club is None and pred is None
