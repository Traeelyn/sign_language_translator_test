Tasks

Run code (unofficial)
cd backend
run prepare_data.py

Completed

- 29 clasess (includes a-z, and a few basic signs)
- feature extraction for live detector setup - includes drawing of hand
- Live detector can detect motion and no motion
- Live detector can detect 2 hands simultaneously

[Video Frame] ---> [feature_extractor.extract_frame_features]
|
v
[frame feature vector: 270]
|
[prepare_data.py] Collects these for each frame → makes sequences
|
v
[X.npy shape = (#videos, 30, 270)]
[y.npy shape = (#videos, #classes)]
