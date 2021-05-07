# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

mkdir -p tirg-models
wget https://storage.googleapis.com/image_retrieval_css/pretrained_models/checkpoint_css3d.pth -O tirg-models/checkpoint_css3d.pth
wget https://storage.googleapis.com/image_retrieval_css/pretrained_models/checkpoint_fashion200k.pth -O tirg-models/checkpoint_fashion200k.pth
wget https://storage.googleapis.com/image_retrieval_css/pretrained_models/checkpoint_mitstates.pth -O tirg-models/checkpoint_mitstates.pth


mkdir -p tirg-data
wget https://storage.googleapis.com/image_retrieval_css/CSS.zip -O tirg-data/CSS.zip
wget https://storage.googleapis.com/image_retrieval_css/test_queries.txt -O tirg-data/test_queries.txt

