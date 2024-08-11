# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""OWL v2 CLIP B/16 config."""
from owl_vit.configs import ConfigDict
from owl_vit.checkpoints import load_checkpoint

import os

def get_config():
  """Returns the configuration for text-query-based detection using OWL-ViT."""
  config = ConfigDict()
  config.experiment_name = 'owl_vit_detection'

  # Dataset.
  config.dataset_name = 'owl_vit'
  config.dataset_configs = ConfigDict()
  config.dataset_configs.input_size = 960
  config.dataset_configs.input_range = None
  config.dataset_configs.max_query_length = 16

  # Model.
  config.model_name = 'text_zero_shot_detection'

  config.model = ConfigDict()
  config.model.normalize = True

  config.model.body = ConfigDict()
  config.model.body.type = 'clip'
  config.model.body.variant = 'vit_b16'
  config.model.body.merge_class_token = 'mul-ln'
  config.model.box_bias = 'both'

  # Objectness head.
  config.model.objectness_head = ConfigDict()
  config.model.objectness_head.stop_gradient = True

  # Init.
  config.init_from = ConfigDict()
  checkpoint_path = load_checkpoint('owl_v2_b16')

  if checkpoint_path is None:
    raise ValueError('Unknown init_mode: {}'.format(init_mode))
  config.init_from.checkpoint_path = checkpoint_path

  return config
