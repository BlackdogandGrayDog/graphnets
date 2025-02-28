# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Commonly used data structures and functions."""

import enum
import tensorflow.compat.v1 as tf
import numpy as np

# Simulator Intrinsic Camera Matrix
K_simulator = np.array([
    [512, 0, 256],
    [0, 512, 256],
    [0, 0, 1]
], dtype=np.float32)

# Hamlyn Intrinsic Camera Matrix
K_hamlyn = np.array([
    [579.05693, 0, 139.93160057],
    [0, 579.05693, 159.01899052],
    [0, 0, 1]
], dtype=np.float32)


extrinsic = np.array([
    [0.9990, -0.0112, -0.0426, -5.49238],
    [0.0117,  0.9999,  0.0097,  0.04267],
    [0.0425, -0.0102,  0.9990, -0.39886],
    [0, 0, 0, 1]  # Add homogeneous row
], dtype=np.float32)

class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 9
  PATCH_SIZE = 10


def triangles_to_edges(faces):
  """Computes mesh edges from triangles."""
  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  # remove duplicates and unpack
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))
