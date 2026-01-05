import os
import json
import timm
import torch
import numpy as np
import pandas as pd
import comfy.utils
import folder_paths
from PIL import Image
from huggingface_hub import hf_hub_download


class AnimeTimmNode:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.tags_df = None

    @classmethod
    def INPUT_TYPES(s):
        models_folder = os.path.join(folder_paths.models_dir, "animetimm")
        if not os.path.exists(models_folder):
            os.makedirs(models_folder, exist_ok=True)

        available_models = [
            "animetimm/caformer_b36.dbv4-full",
            "animetimm/caformer_m36.dbv4-full",
            "animetimm/caformer_s18.dbv4-full",
            "animetimm/caformer_s36.dbv4-full",
            "animetimm/convnextv2_huge.dbv4-full",
            "animetimm/convnext_base.dbv4-full",
            "animetimm/eva02_large_patch14_448.dbv4-full",
            "animetimm/mobilenetv3_large_100.dbv4-full",
            "animetimm/mobilenetv3_large_150d.dbv4-full",
            "animetimm/mobilenetv4_conv_aa_large.dbv4-full",
            "animetimm/mobilenetv4_conv_small.dbv4-full",
            "animetimm/mobilenetv4_conv_small_050.dbv4-full",
            "animetimm/resnet101.dbv4-full",
            "animetimm/resnet152.dbv4-full",
            "animetimm/resnet18.dbv4-full",
            "animetimm/resnet34.dbv4-full",
            "animetimm/resnet50.dbv4-full",
            "animetimm/swinv2_base_window8_256.dbv4-full",
            "animetimm/swinv2_base_window8_256.dbv4a-full",
            "animetimm/vit_base_patch16_224.dbv4-full",
        ]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image for tag prediction"}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Threshold for tag prediction confidence",
                    },
                ),
                "model_repo": (
                    available_models,
                    {
                        "default": "animetimm/caformer_s36.dbv4-full",
                        "tooltip": "Model to use for tag prediction",
                    },
                ),
                "include_general": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Include general tags in output",
                    },
                ),
                "include_character": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Include character tags in output",
                    },
                ),
                "include_artist": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include artist tags in output",
                    },
                ),
                "include_rating": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Include rating tags in output",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "tags",
        "confidence_scores",
        "raw_output",
        "general_tags",
        "character_tags",
        "artist_tags",
        "rating_tags",
    )
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    FUNCTION = "predict"
    CATEGORY = "AnimeTimm"
    DESCRIPTION = "Predict anime tags from images using timm models"

    def predict(
        self,
        image,
        threshold,
        model_repo,
        include_general,
        include_character,
        include_artist,
        include_rating,
    ):
        batch_results = []
        formatted_tags_list = []
        all_scores_list = []
        raw_output_list = []
        general_str_list = []
        character_str_list = []
        artist_str_list = []
        rating_str_list = []
        pbar = comfy.utils.ProgressBar(image.shape[0])

        for i in range(image.shape[0]):
            i_img = 255.0 * image[i].cpu().numpy()
            img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))

            try:
                self.model_repo = model_repo
                if (
                    self.model is None
                    or getattr(self, "model_repo_loaded", "") != model_repo
                ):
                    self._load_model_auto(model_repo)
                    self.model_repo_loaded = model_repo
            except Exception as e:
                print(f"Error loading model: {e}")
                formatted_tags_list.append("")
                all_scores_list.append([])
                raw_output_list.append("")
                general_str_list.append("")
                character_str_list.append("")
                artist_str_list.append("")
                rating_str_list.append("")
                pbar.update(1)
                continue

            input_tensor = self.preprocessor(img).unsqueeze(0)

            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output)[0].cpu().numpy()

            general_tags = []
            character_tags = []
            artist_tags = []
            rating_tags = []

            for idx, (tag_row, score) in enumerate(
                zip(self.tags_df.itertuples(), prediction)
            ):
                tag_name = tag_row.name
                category = tag_row.category
                best_threshold = tag_row.best_threshold

                effective_threshold = max(threshold, best_threshold)

                if score >= effective_threshold:
                    if category == 0:  # General
                        if include_general:
                            general_tags.append((tag_name, score))
                    elif category == 4:  # Character
                        if include_character:
                            character_tags.append((tag_name, score))
                    elif category == 1:  # Artist
                        if include_artist:
                            artist_tags.append((tag_name, score))
                    elif category == 9:  # Rating
                        if include_rating:
                            rating_tags.append((tag_name, score))

            all_tags = []
            all_scores = []

            if include_general:
                all_tags.extend([tag for tag, score in general_tags])
                all_scores.extend([float(score) for tag, score in general_tags])

            if include_character:
                all_tags.extend([tag for tag, score in character_tags])
                all_scores.extend([float(score) for tag, score in character_tags])

            if include_artist:
                all_tags.extend([tag for tag, score in artist_tags])
                all_scores.extend([float(score) for tag, score in artist_tags])

            if include_rating:
                all_tags.extend([tag for tag, score in rating_tags])
                all_scores.extend([float(score) for tag, score in rating_tags])

            formatted_tags = ", ".join(all_tags)
            formatted_tags_list.append(formatted_tags)
            all_scores_list.append(all_scores)

            raw_output_parts = []
            if include_general:
                for tag, score in general_tags:
                    raw_output_parts.append(f"general: {tag}: {score:.3f}")
            if include_character:
                for tag, score in character_tags:
                    raw_output_parts.append(f"character: {tag}: {score:.3f}")
            if include_artist:
                for tag, score in artist_tags:
                    raw_output_parts.append(f"artist: {tag}: {score:.3f}")
            if include_rating:
                for tag, score in rating_tags:
                    raw_output_parts.append(f"rating: {tag}: {score:.3f}")

            raw_output = "\n".join(raw_output_parts)
            raw_output_list.append(raw_output)

            general_str = ", ".join([tag for tag, score in general_tags])
            character_str = ", ".join([tag for tag, score in character_tags])
            artist_str = ", ".join([tag for tag, score in artist_tags])
            rating_str = ", ".join([tag for tag, score in rating_tags])

            general_str_list.append(general_str)
            character_str_list.append(character_str)
            artist_str_list.append(artist_str)
            rating_str_list.append(rating_str)

            batch_results.append(
                (
                    formatted_tags,
                    all_scores,
                    raw_output,
                    general_str,
                    character_str,
                    artist_str,
                    rating_str,
                )
            )

            pbar.update(1)

        return {
            "ui": {"info": formatted_tags_list},
            "result": (
                formatted_tags_list,
                all_scores_list,
                raw_output_list,
                general_str_list,
                character_str_list,
                artist_str_list,
                rating_str_list,
            ),
        }

    def _load_model_auto(self, model_repo):
        """检测本地模型，如果不存在则从Hugging Face下载"""
        model_folder_name = model_repo.split("/", 1)[-1]
        local_model_root = os.path.join(folder_paths.models_dir, "animetimm")

        os.makedirs(local_model_root, exist_ok=True)

        local_model_path = os.path.join(local_model_root, model_folder_name)

        required_files = [
            "config.json",
            "preprocess.json",
            "selected_tags.csv",
            "pytorch_model.bin",
        ]
        local_model_exists = all(
            [os.path.exists(os.path.join(local_model_path, f)) for f in required_files]
        )

        if local_model_exists:
            print(f"Loading model from local path: {local_model_path}")
            self._load_model_local(local_model_path)
        else:
            print(
                f"Local model not found at {local_model_path}, downloading from HuggingFace..."
            )
            self._download_and_save_model(
                model_repo, local_model_root, model_folder_name
            )
            self._load_model_local(local_model_path)

    def _download_and_save_model(self, model_repo, local_model_root, model_folder_name):
        """从Hugging Face下载模型并保存到本地"""
        print(f"Downloading model {model_repo} to {local_model_root}")

        actual_model_path = os.path.join(local_model_root, model_folder_name)

        required_files = [
            "config.json",
            "preprocess.json",
            "selected_tags.csv",
            "pytorch_model.bin",
        ]

        try:
            for filename in required_files:
                hf_hub_download(
                    repo_id=model_repo,
                    filename=filename,
                    local_dir=actual_model_path,
                    local_dir_use_symlinks=False,
                )
            print(f"Model downloaded successfully to {actual_model_path}")
        except Exception as e:
            print(f"Failed to download from original repo: {e}")
            print(f"Trying backup repo: Makki2104/animetimm")

            model_name = model_repo.split("/")[-1]
            backup_repo = f"Makki2104/animetimm"

            os.makedirs(actual_model_path, exist_ok=True)

            for filename in required_files:
                hf_hub_download(
                    repo_id=backup_repo,
                    filename=f"{model_name}/{filename}",
                    local_dir=local_model_root,
                    local_dir_use_symlinks=False,
                )

            print(
                f"Model downloaded successfully from backup repo to {actual_model_path}"
            )

    def _load_model_local(self, local_model_path):
        """从本地路径加载模型"""
        print(f"Loading model from local path: {local_model_path}")

        self._validate_model_path(local_model_path)

        local_model_identifier = f"local-dir:{local_model_path}"

        self.model = timm.create_model(local_model_identifier, pretrained=True)
        self.model.eval()

        self._load_preprocessor_and_tags(local_model_path)

        print(f"Model and tags loaded successfully from local path: {local_model_path}")

    def _validate_model_path(self, local_model_path):
        """验证模型路径是否存在及必需文件是否齐全"""
        if not os.path.exists(local_model_path):
            raise Exception(f"Local model path does not exist: {local_model_path}")

        required_files = [
            "config.json",
            "preprocess.json",
            "selected_tags.csv",
            "pytorch_model.bin",
        ]
        for file in required_files:
            file_path = os.path.join(local_model_path, file)
            if not os.path.exists(file_path):
                raise Exception(
                    f"Required file {file} not found in local path: {file_path}"
                )

    def _load_preprocessor_and_tags(self, local_model_path):
        """加载预处理器和标签"""
        from imgutils.preprocess import create_torchvision_transforms

        preprocess_file = os.path.join(local_model_path, "preprocess.json")
        with open(preprocess_file, "r") as f:
            preprocessor_config = json.load(f)["test"]
        self.preprocessor = create_torchvision_transforms(preprocessor_config)

        tags_file = os.path.join(local_model_path, "selected_tags.csv")
        self.tags_df = pd.read_csv(tags_file, keep_default_na=False)


NODE_CLASS_MAPPINGS = {"AnimeTimmNode": AnimeTimmNode}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimeTimmNode": "Anime TIMM Classifier"}
