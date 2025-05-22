# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=redefined-outer-name,missing-module-docstring,g-importing-member,missing-function-docstring,g-bare-generic
from argparse import ArgumentParser
import os
import pickle
from typing import Dict
from data_utils import build_circo_dataset
from data_utils import build_fiq_dataset
from flax import serialization
import jax
import jax.numpy as jnp
from model import MagicLens
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from tqdm import tqdm


def load_model(model_size: str, model_path: str) -> Dict:
    # init model
    model = MagicLens(model_size)
    rng = jax.random.PRNGKey(0)
    dummpy_input = {
        "ids": jnp.ones((1, 1, 77), dtype=jnp.int32),
        "image": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
    }
    params = model.init(rng, dummpy_input)
    print("model initialized")
    # load model
    with open(model_path, "rb") as f:
        model_bytes = pickle.load(f)
    params = serialization.from_bytes(params, model_bytes)
    print("model loaded")
    return model, params


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./magic_lens_clip_base.pkl",
        help="The path to model directory.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        help="Model size, choices: base, large.",
        choices=["base", "large"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fiq-dress",
        help="Dataset selection.",
        choices=["fiq-dress", "fiq-shirt", "fiq-toptee", "circo", "dtin"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory of predictions top 50.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for inference."
    )
    args = parser.parse_args()

    # init model
    tokenizer = clip_tokenizer.build_tokenizer()
    model, model_params = load_model(args.model_size, args.model_path)

    # load data
    if args.dataset.startswith("fiq"):
        subtask = args.dataset.split("-")[1]
        eval_dataset = build_fiq_dataset(dataset_name=args.dataset, tokenizer=tokenizer)
    elif args.dataset in ["circo"]:
        eval_dataset = build_circo_dataset(
            dataset_name=args.dataset, tokenizer=tokenizer
        )
    else:
        raise NotImplementedError

    # inference index:
    index_embeddings = []
    print("Inference index...")
    for i in tqdm(range(0, len(eval_dataset.index_examples), args.batch_size)):
        batch = eval_dataset.index_examples[i : i + args.batch_size]
        iids = [i.iid for i in batch]
        iimages = jnp.concatenate([i.iimage for i in batch], axis=0)
        itokens = jnp.concatenate([i.itokens for i in batch], axis=0)
        iembeds = model.apply(model_params, {"ids": itokens, "image": iimages})[
            "multimodal_embed_norm"
        ]
        index_embeddings.append(iembeds)

    index_embeddings = jnp.concatenate(index_embeddings, axis=0)

    print("Inference queries...")
    for i in tqdm(range(0, len(eval_dataset.query_examples), args.batch_size)):
        batch = eval_dataset.query_examples[i : i + args.batch_size]
        qids = [q.qid for q in batch]
        qimages = jnp.concatenate([q.qimage for q in batch], axis=0)
        qtokens = jnp.concatenate([q.qtokens for q in batch], axis=0)
        qembeds = model.apply(model_params, {"ids": qtokens, "image": qimages})[
            "multimodal_embed_norm"
        ]
        similarity_scores = jnp.dot(qembeds, index_embeddings.T)
        # get top 50 by similarity (by default)
        top_k_indices = jnp.argsort(similarity_scores, axis=1)[:, -50:][:, ::-1]
        top_k_iids = [
            [eval_dataset.index_examples[idx].iid for idx in top_k]
            for top_k in top_k_indices
        ]

        # gather scores for the top_k
        top_k_scores = [
            similarity_scores[i, tk].tolist() for i, tk in enumerate(top_k_indices)
        ]

        # update the query_example with the retrieved results
        for k, q_example in enumerate(batch):
            q_example.retrieved_iids = top_k_iids[k]
            q_example.retrieved_scores = top_k_scores[k]
            eval_dataset.query_examples[i + k] = q_example
    # Post-processing and evaluation:
    if args.dataset in ["fiq-dress", "fiq-shirt", "fiq-toptee"]:
        eval_dataset.evaluate_recall()
    elif args.dataset in ["circo"]:
        eval_dataset.write_to_file(
            os.path.join(args.output, args.dataset + "_" + args.model_size)
        )
    else:
        raise NotImplementedError

    print("Inference Done.")
