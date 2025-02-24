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
"""Runs the learner/evaluator."""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
import threading
import time
import matplotlib.pyplot as plt
# import tensorflow.compat.v1 as tf
from meshgraphnets import cfd_eval
from meshgraphnets import cfd_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import core_model
from meshgraphnets import dataset




FLAGS = flags.FLAGS

# Trajectory and loss settings
trajectory = '6'
loss_model = 'patched_0.01_0.005'
steps = '56k'

# Base directory
base_dir = f'./meshgraphnets/dataset/'
os.makedirs(base_dir, exist_ok=True)

# Evaluation file name
eval_file_name = f'gtd_input_data_7_ts_21_gs_19_cotracker_accel'

# Mode and model selection
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Train model or run evaluation.')
flags.DEFINE_enum('model', 'cloth', ['cfd', 'cloth'], 'Select model to run.')

# Loss model flag
flags.DEFINE_string('loss_model', loss_model, 'Loss model type (e.g., orig or patched_xxx)')

# Checkpoint directory
flags.DEFINE_string('checkpoint_dir', f'{base_dir}{loss_model}_loss_{steps}_ckpts/', 'Directory to save checkpoints')

# Rollout file path
flags.DEFINE_string('rollout_path', f'{base_dir}eval/{eval_file_name}.pkl', 'Pickle file to save evaluation trajectories')

# Additional settings
flags.DEFINE_integer('num_rollouts', 1, 'Number of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'Number of training steps')

PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}

# Function to handle real-time plottingS
def real_time_plot(steps, train_losses, val_steps, val_losses, stop_event, checkpoint_dir, data_lock):
    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots(figsize=(12, 6))
    zoom_ax = ax.twinx()  # Add a secondary y-axis for zoomed-in view

    # Initialize primary plot
    train_line, = ax.plot([], [], label="Train Loss", color="blue", linewidth=3, alpha=0.7)  # Bold blue line
    val_line, = ax.plot([], [], label="Validation Loss", color="red", linestyle="-", linewidth=2, alpha=0.5)  # Red solid line
    zoom_train_line, = zoom_ax.plot([], [], label="Zoomed Train Loss", color="orange", linestyle="--", linewidth=1.5)
    zoom_val_line, = zoom_ax.plot([], [], label="Zoomed Validation Loss", color="green", linestyle="--", linewidth=1.5)  # Green dashed line

    # Styling the primary plot
    ax.set_xlabel("Global Step", fontsize=14)
    ax.set_ylabel("Loss (Full Scale)", fontsize=14, color="blue")
    zoom_ax.set_ylabel("Loss (Zoomed)", fontsize=14, color="orange")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_title("Real-Time Training and Validation Loss", fontsize=16, fontweight="bold")
    fig.tight_layout()

    # Create legend combining all axes
    lines = [train_line, val_line, zoom_train_line, zoom_val_line]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", fontsize=12)

    try:
        while not stop_event.is_set():
            if steps or val_steps:
                with data_lock:
                    # Update primary plot for training loss
                    train_line.set_xdata(steps)
                    train_line.set_ydata(train_losses)
                    ax.relim()
                    ax.autoscale_view()

                    # Update validation loss plot
                    val_line.set_xdata(val_steps)
                    val_line.set_ydata(val_losses)

                    # Update zoomed-in plot for training loss with losses ≤ 0.1
                    zoomed_train_losses = [loss for step, loss in zip(steps, train_losses) if loss <= 0.1]
                    zoomed_train_steps = [step for step, loss in zip(steps, train_losses) if loss <= 0.1]
                    zoom_train_line.set_xdata(zoomed_train_steps)
                    zoom_train_line.set_ydata(zoomed_train_losses)

                    # Update zoomed-in plot for validation loss with losses ≤ 0.1
                    zoomed_val_losses = [loss for step, loss in zip(val_steps, val_losses) if loss <= 0.1]
                    zoomed_val_steps = [step for step, loss in zip(val_steps, val_losses) if loss <= 0.1]
                    zoom_val_line.set_xdata(zoomed_val_steps)
                    zoom_val_line.set_ydata(zoomed_val_losses)

                    # Set zoomed-in axis limits
                    zoom_ax.set_ylim(0, 0.1)

                    ax.relim()
                    zoom_ax.relim()
                    ax.autoscale_view()
                    zoom_ax.autoscale_view()

                plt.pause(0.1)  # Refresh plot
            time.sleep(0.1)  # Prevent excessive updates

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving plot...")

    finally:
        # Save the final plot when training ends or interrupted
        plt.ioff()
        plot_path = os.path.join(checkpoint_dir, "training_validation_loss_plot.png")
        fig.savefig(plot_path, dpi=300)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)

#  Learner Function
def learner(model, params):
    """Run a learner job with real-time visualization (no validation)."""
    # Shared lists for steps and training losses
    steps = []
    train_losses = []
    stop_event = threading.Event()  # Event to signal thread to stop
    data_lock = threading.Lock()  # Lock to prevent data asynchrony

    # Start the real-time plot in a separate thread
    plot_thread = threading.Thread(target=real_time_plot, args=(steps, train_losses, [], [], stop_event, FLAGS.checkpoint_dir, data_lock))
    plot_thread.start()

    try:
        # Dataset preparation
        train_ds = dataset.load_datasets('./meshgraphnets/dataset/trajectory_' + trajectory + '_accel_patched_input/')
        train_ds = train_ds.flat_map(tf.data.Dataset.from_tensor_slices).shuffle(10000).repeat(None)
        train_inputs = tf.data.make_one_shot_iterator(train_ds).get_next()
        
        # Loss and optimizer
        if FLAGS.loss_model == 'orig':
            loss_op = model.loss_orig(train_inputs)
        else:
            loss_op = model.loss(train_inputs)
        
        global_step = tf.train.create_global_step()
        lr = tf.train.exponential_decay(learning_rate=1e-4,
                                        global_step=global_step,
                                        decay_steps=int(5e6),
                                        decay_rate=0.1) + 1e-8
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss_op, global_step=global_step)

        # Adjust training operation for warm-up
        train_op = tf.cond(tf.less(global_step, 100),
                           lambda: tf.group(tf.assign_add(global_step, 1)),
                           lambda: tf.group(train_op))

        # Training session
        with tf.train.MonitoredTrainingSession(
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)],
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_checkpoint_secs=100) as sess:

            while not sess.should_stop():
                _, step, train_loss = sess.run([train_op, global_step, loss_op])

                if step % 10 == 0:
                    logging.info('Step %d: Train Loss %g', step, train_loss)
                    with data_lock:
                        steps.append(step)
                        train_losses.append(train_loss)  # Append training loss for the plot

            logging.info('Training complete.')

    except KeyboardInterrupt:
        logging.warning("Training interrupted by user. Finalizing and saving the plot...")

    finally:
        # Signal the plot thread to stop and wait for it to finish
        stop_event.set()
        plot_thread.join()


def evaluator(model, params):
  """Run a model rollout trajectory."""
  ds = dataset.convert_to_tf_dataset('./input/trajectory_' + trajectory + '_accel_patched_eval/' + eval_file_name + '.npz')
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs)
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    trajectories = []
    scalars = []
    for traj_idx in range(FLAGS.num_rollouts):
      logging.info('Rollout trajectory %d', traj_idx)
      scalar_data, traj_data = sess.run([scalar_op, traj_ops])
      trajectories.append(traj_data)
      scalars.append(scalar_data)
    for key in scalars[0]:
      logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    with open(FLAGS.rollout_path, 'wb') as fp:
      pickle.dump(trajectories, fp)

    
    
def main(argv):
  del argv
  tf.enable_resource_variables()
  tf.disable_eager_execution()
  params = PARAMETERS[FLAGS.model]
  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=15)
  model = params['model'].Model(learned_model, loss_model)
  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)

if __name__ == '__main__':
  app.run(main)
