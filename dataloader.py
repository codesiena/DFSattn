import json
import os
import re
import unicodedata


def prompt_folder_name(prompt_idx, prompt, max_prompt_chars=56):
    """Return a filesystem-safe folder name containing prompt id and text."""
    prompt = unicodedata.normalize("NFKD", str(prompt).strip())
    prompt = prompt.encode("ascii", "ignore").decode("ascii").lower()
    prompt_slug = re.sub(r"[^a-z0-9]+", "_", prompt).strip("_")
    prompt_slug = prompt_slug[:max_prompt_chars].rstrip("_")
    prefix = f"prompt_{int(prompt_idx):03d}"
    return f"{prefix}_{prompt_slug}" if prompt_slug else prefix


def load_prompt(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list


def load_image(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    image_list = [prompt["image_path"] for prompt in prompt_list]
    return image_list


def load_prompt_or_image(prompt_source, prompt_idx, prompt, image_path):
    """
    Load the prompt or image path based on the prompt source.
    """
    if prompt_source == "prompt":
        assert prompt_idx == 0, "You have already provided a prompt"
        return prompt, image_path
    elif prompt_source == "I2V_VBench":
        # assert prompt is a json file
        assert prompt.endswith(".json"), "Prompt must be a json file"
        with open(prompt, "r") as f:
            prompts = json.load(f)

        prompt_idx = str(prompt_idx)
        original_prompt = prompts[prompt_idx]["original"]
        improved_prompt = prompts[prompt_idx]["improved"]
        image_path = os.path.join(image_path, f"{original_prompt}.jpg")
        assert os.path.exists(image_path), "Image path does not exist"

        return improved_prompt, image_path
    elif prompt_source == "I2V_Wan_Web":
        assert prompt == image_path, "Prompt and image path must be the same"

        prompt_idx = str(prompt_idx).zfill(3)
        prompt_path = os.path.join(prompt, f"{prompt_idx}/prompt.txt")
        image_path = os.path.join(image_path, f"{prompt_idx}/image.jpg")

        with open(prompt_path, "r") as f:
            prompt = f.read()
        return prompt, image_path

    elif prompt_source in ["T2V_Wan_VBench", "T2V_Hyv_VBench", "T2V_Hyv_Web"]:
        assert prompt.endswith(".txt"), "Prompt must be a txt file"
        with open(prompt, "r") as f:
            prompts = f.readlines()

        prompt = prompts[prompt_idx]
        return prompt, None
    elif prompt_source in ["T2V_Xingyang_Motion", "T2V_Xingyang_VBench"]:
        assert prompt.endswith(".txt"), "Prompt must be a txt file"
        with open(prompt, "r") as f:
            prompts = f.readlines()

        prompt = prompts[prompt_idx]
        return prompt, None
    else:
        raise ValueError(f"Invalid prompt source: {prompt_source}")
