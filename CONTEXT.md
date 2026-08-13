# Football-Analysis domain language

- **Camera Profile**: One physical camera/lens/focal setting at a fixed resolution, described by an intrinsic matrix and distortion coefficients.
- **Pitch Anchor Set**: Image-to-pitch correspondences selected for one continuous video shot. Anchors initialize absolute camera geometry.
- **Camera Pose**: The per-frame rotation and translation that map metric pitch coordinates into the camera image, plus its quality and uncertainty.
- **Pitch Projection**: A mapping between image pixels and the metric ground plane. It is valid only for points on or very near the pitch.
- **Ball Candidate**: One detector observation that may be a football. It is not a Ball Track until temporal evidence confirms it.
- **Ball Track**: A temporally associated sequence of Ball Candidates with an explicit tentative, confirmed, occluded, ambiguous, or rejected state.
- **Ball Track Segment**: A continuous part of a Ball Track that does not cross a camera cut, kick, impact, bounce, or irrecoverable observation gap.
- **Ball Kinematics**: A ground or airborne physical-motion estimate produced from a confirmed Ball Track and reliable Camera Poses.
- **Reliable Ball Speed**: A speed value whose observation span, pose quality, trajectory residual, and uncertainty all pass the configured quality policy.
- **Diagnostic Reason**: A stable machine-readable explanation for why a Camera Pose, Ball Track, or Ball Kinematics result is not reliable.
