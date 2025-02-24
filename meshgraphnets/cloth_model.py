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
"""Model for FlagSimple."""

import sonnet as snt
import tensorflow.compat.v1 as tf
from meshgraphnets import common
from meshgraphnets import core_model
from meshgraphnets import normalization
from absl import flags
from absl import logging
FLAGS = flags.FLAGS


class Model(snt.AbstractModule):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, loss_model, name='Model'):
    super(Model, self).__init__(name=name)
    with self._enter_variable_scope():
        self._learned_model = learned_model
        self._output_normalizer = normalization.Normalizer(
            size=3, name='output_normalizer'
        )

        if loss_model == 'orig':
            # Original Node Normalizer
            self._node_normalizer = normalization.Normalizer(
                size=3 + common.NodeType.SIZE, name='node_normalizer'
            )
        else:
            # Patched Node Normalizer (for any 'patched_xxx' variant)
            self._node_normalizer = normalization.Normalizer(
                size=3 + 3 + 2 + common.NodeType.SIZE + common.NodeType.PATCH_SIZE ** 2 * 2,
                name='node_normalizer'
            )

        self._edge_normalizer = normalization.Normalizer(
            size=7, name='edge_normalizer'  # 2D coord + 3D coord + 2*length = 7
        )

  #  With Patch and Acceleration Input
  def _build_graph(self, inputs, is_training):
    """Builds input graph."""
    # construct graph nodes
    logging.info('Building patched graph')
    # Compute velocity and acceleration
    velocity = inputs['world_pos'] - inputs['prev|world_pos']
    acceleration = inputs['world_pos'] - 2 * inputs['prev|world_pos'] + inputs['prev_prev|world_pos']
    
    # Compute 2D optical flow for feature points
    flow_2d = inputs['image_pos'] - inputs['prev|image_pos']
    
    # Flatten local patched optical flow
    patched_flow = tf.reshape(inputs['patched_flow'], [tf.shape(inputs['patched_flow'])[0], -1])  # (num_nodes, 200)
    
    # One-hot encode node type
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    # Concatenate all node features
    node_features = tf.concat([velocity, acceleration, flow_2d, patched_flow, node_type], axis=-1)

    # construct graph edges
    senders, receivers = common.triangles_to_edges(inputs['cells'])
    relative_world_pos = (tf.gather(inputs['world_pos'], senders) -
                          tf.gather(inputs['world_pos'], receivers))
    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))
    edge_features = tf.concat([
        relative_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)
    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])
  
  #  Original Graph
  def _build_graph_orig(self, inputs, is_training):
    """Builds input graph."""
    # construct graph nodes
    logging.info('Building original graph')
    velocity = inputs['world_pos'] - inputs['prev|world_pos']
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_features = tf.concat([velocity, node_type], axis=-1)

    # construct graph edges
    senders, receivers = common.triangles_to_edges(inputs['cells'])
    relative_world_pos = (tf.gather(inputs['world_pos'], senders) -
                          tf.gather(inputs['world_pos'], receivers))
    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))
    edge_features = tf.concat([
        relative_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)
    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])


  def _build(self, inputs):
    if FLAGS.loss_model == 'orig':  # Fetch loss model from FLAGS
        graph = self._build_graph_orig(inputs, is_training=False)
    else:
        graph = self._build_graph(inputs, is_training=False)

    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output)

  #  Original Loss Function
  @snt.reuse_variables
  def loss_orig(self, inputs):
    """L2 loss on position."""
    graph = self._build_graph_orig(inputs, is_training=True)
    network_output = self._learned_model(graph)

    # build target acceleration
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    target_position = inputs['target|world_pos']
    target_acceleration = target_position - 2*cur_position + prev_position
    target_normalized = self._output_normalizer(target_acceleration)

    # build loss
    loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
    loss = tf.reduce_mean(error[loss_mask])
    return loss
  
  # Loss Function with Acceleration Regularization
  @snt.reuse_variables
  def loss(self, inputs):
      """L2 loss on position with acceleration regularization."""
      logging.info('Patched loss')
      logging.info('Patched loss')
      logging.info('Patched loss')
      logging.info('Patched loss')
      logging.info('Patched loss')
      logging.info('Patched loss')
      logging.info('Patched loss')
      logging.info('Patched loss')
      graph = self._build_graph(inputs, is_training=True)
      network_output = self._learned_model(graph)

      # Compute target acceleration
      cur_position = inputs['world_pos']
      prev_position = inputs['prev|world_pos']
      prev2_position = inputs['prev_prev|world_pos']
      target_position = inputs['target|world_pos']
      
      target_acceleration = target_position - 2 * cur_position + prev_position
      current_acceleration = cur_position - 2 * prev_position + prev2_position

      target_normalized = self._output_normalizer(target_acceleration)
      current_normalized = self._output_normalizer(current_acceleration)

      # Base L2 loss on acceleration prediction
      loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
      error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
      base_loss = tf.reduce_mean(error[loss_mask])

      # Acceleration regularization: || A_t+1 - A_t ||^2
      acceleration_reg = tf.reduce_mean(tf.square(current_normalized - network_output))
      
      # Denormalize network output before using it
      acceleration = self._output_normalizer.inverse(network_output)
      # Convert acceleration to predicted position
      pred_position = 2 * cur_position + acceleration - prev_position
      pred_position_homo = tf.concat([pred_position, tf.ones_like(pred_position[:, :1])], axis=-1)

      # Project to 2D using camera matrix
      image_pos_pred = tf.matmul(common.K, tf.transpose(pred_position_homo[:, :3], perm=[1, 0]))
      image_pos_pred = tf.transpose(image_pos_pred, perm=[1, 0])
      image_pos_pred = image_pos_pred[:, :2] / image_pos_pred[:, 2:3]
      
    #   print_op = tf.print("Predicted image Position:\n", image_pos_pred, 
    #                 "\nTarget image Position:\n", inputs['target|image_pos'])
      
      # Patch Regularization Loss: || pred_flow - target_patched_flow ||^2
      pred_flow = image_pos_pred[:, :2] - inputs['image_pos']
      # Patch Regularization Loss using the **averaged patch flow**
      patch_reg = tf.reduce_mean(tf.square(pred_flow - inputs['avg_target|patched_flow']))
      
      # # Compute L2 loss on position
      # pred_normalized = self._output_normalizer(pred_position)
      # target_normalized = self._output_normalizer(target_position)
      # error = tf.reduce_sum((pred_normalized - target_normalized)**2, axis=1)
      # base_loss = tf.reduce_mean(error[loss_mask])

      # Final loss with regularization
      loss_model_values = FLAGS.loss_model.replace("patched_", "").split("_")
      if len(loss_model_values) == 1:
          lambda_accel = lambda_patch = float(loss_model_values[0])  # Both are the same
      elif len(loss_model_values) == 2:
          lambda_accel, lambda_patch = map(float, loss_model_values)  # Extract both separately
      else:
          raise ValueError(f"Invalid loss_model format: {FLAGS.loss_model}")
      
      total_loss = base_loss + lambda_accel * acceleration_reg + lambda_patch * patch_reg
      logging.info(f"Acceleration regularization: {acceleration_reg}")
      logging.info(f"Patch regularization: {patch_reg}")
      
      return total_loss


  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs.""" 
    acceleration = self._output_normalizer.inverse(per_node_network_output)
    # integrate forward
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    position = 2*cur_position + acceleration - prev_position
    return position
