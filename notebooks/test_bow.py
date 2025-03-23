# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import cv2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from project.foundation.dbow import BowDatabase, BowVocabulary
from project.utils.features import orb_detect_and_compute
from project.utils.paths import repo_root

# %%
vid_path = repo_root() / "data/monumental_take_home/plantage_shed.mp4"

# %%
train_frames = []
test_frames = []
for i, frame in tqdm(enumerate(iio.imiter(vid_path))):

    if i % 50 == 0:
        train_frames.append(frame)
    if i % 63 == 0:
        test_frames.append(frame)

    

# %%
orb = cv2.ORB.create()

descriptors = []
for frame in train_frames:
    _, desc = orb_detect_and_compute(frame, orb)
    descriptors.append(desc.astype(np.float32))

# %%
vocab = BowVocabulary()

# %%
descriptors[0].shape, descriptors[0].dtype

# %%
vocab.create(descriptors)

# %%
vocab.info()

# %%
vocab_path = repo_root() / "data/bow_vocab/vocab.vocab"
vocab_path.parent.mkdir(exist_ok=True, parents=True)
vocab.save(vocab_path)

# %%
vocab = BowVocabulary()
vocab.load(vocab_path)

# %%
vocab.info()

# %%
new_descriptors = []
for frame in test_frames:
    _, desc = orb_detect_and_compute(frame, orb)
    new_descriptors.append(desc.astype(np.float32))

# %%
from itertools import combinations

bowvecs = [vocab.transform(d) for d in new_descriptors]

for (i1, d1), (i2, d2) in combinations(enumerate(bowvecs), r=2):
    print(f"i1-i2 score: {vocab.score(d1, d2)}")

# %%
db = BowDatabase(vocab)
db.info()

# %%
for desc in new_descriptors:
    db.add(desc)

# %%
db.info()

# %%
db.query(new_descriptors[0], 5)
