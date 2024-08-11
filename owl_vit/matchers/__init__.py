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

"""Matching utilities for Object Detection models."""

from owl_vit.matchers.common import cpu_matcher
from owl_vit.matchers.common import slicer
from owl_vit.matchers.greedy import greedy_matcher
from owl_vit.matchers.hungarian import hungarian_matcher
from owl_vit.matchers.hungarian_cover import hungarian_cover_tpu_matcher
from owl_vit.matchers.hungarian_jax import hungarian_scan_tpu_matcher
from owl_vit.matchers.hungarian_jax import hungarian_tpu_matcher
from owl_vit.matchers.lazy import lazy_matcher
from owl_vit.matchers.sinkhorn import sinkhorn_matcher
