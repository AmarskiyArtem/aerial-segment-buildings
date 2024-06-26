# Copyright (c) 2024, Artem Amarskiy, Anastasiia Kornilova
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

from src.models.rgbfootprint.drn_c42 import DRN_c42
from src.models.SAM.sam import SAM
from src.models.YOLO.yolo import YOLOseg

from src.datasets.sam import SAM_dataset, SAM_collate_fn
from src.datasets.yolo import YOLO_dataset
from src.datasets.rgbfootprint import RGBfootprint_dataset

from torch.utils.data import DataLoader

from src.utils.evaluator import Evaluator

from argparse import ArgumentParser


def eval(model_name, checkpoint_path, device, data_dir, batch_size, model_type=None):
    model_name = model_name.lower()
    if model_name == "drn_c42":
        model = DRN_c42(checkpoint_path, device)
        dataset = RGBfootprint_dataset(data_dir)
    elif model_name == "sam":
        if model_type is None:
            raise ValueError("model_type must be provided for SAM model")
        model = SAM(model_type, checkpoint_path, device)
        dataset = SAM_dataset(data_dir)
    elif model_name == "yolo":
        model = YOLOseg(checkpoint_path, device)
        dataset = YOLO_dataset(data_dir)
    else:
        raise NotImplementedError

    if model_name == "sam":
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=SAM_collate_fn
        )
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluator = Evaluator(model, loader, device, False)
    metrics = evaluator.evaluate()
    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, choices=["sam", "drn_c42", "yolo"]
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--device", type=str, required=True, help="cuda or cpu")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="the path to the directory with images. Should contains images/ and masks/ dirs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=20, help="batch size (default 20)"
    )
    parser.add_argument(
        "--model_type", type=str, default=None, help="model type for SAM model"
    )
    args = parser.parse_args()
    return eval(
        args.model_name,
        args.checkpoint_path,
        args.device,
        args.data_dir,
        args.batch_size,
        args.model_type,
    )


if __name__ == "__main__":
    print(main())
