import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


STEP_PATTERN = re.compile(r"step_(\d+)$")
LAYER_PATTERN = re.compile(r"layer_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a video that compares one attention head across diffusion "
            "steps while advancing through transformer layers."
        )
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        required=True,
        help="Directory containing step_XXX/layer_YY/block_mask.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output MP4 path",
    )
    parser.add_argument("--head", type=int, default=0, help="Head index to show")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Diffusion steps to compare; default: all discovered steps",
    )
    parser.add_argument("--fps", type=float, default=4.0, help="Output frame rate")
    return parser.parse_args()


def discover_masks(mask_dir):
    masks = {}
    for path in mask_dir.glob("step_*/layer_*/block_mask.npy"):
        step_match = STEP_PATTERN.fullmatch(path.parent.parent.name)
        layer_match = LAYER_PATTERN.fullmatch(path.parent.name)
        if step_match is None or layer_match is None:
            continue
        step = int(step_match.group(1))
        layer = int(layer_match.group(1))
        masks.setdefault(step, {})[layer] = path
    return masks


def load_head(path, head):
    mask = np.load(path, allow_pickle=False)
    if mask.ndim != 3:
        raise ValueError(f"Expected [heads, q_blocks, k_blocks] in {path}, got {mask.shape}")
    if head < 0 or head >= mask.shape[0]:
        raise ValueError(f"Head {head} is out of range for {path} with {mask.shape[0]} heads")
    return mask[head].astype(bool, copy=False)


def main():
    args = parse_args()
    masks = discover_masks(args.mask_dir)
    if not masks:
        raise FileNotFoundError(f"No block_mask.npy files found under {args.mask_dir}")

    steps = sorted(masks) if args.steps is None else args.steps
    missing_steps = [step for step in steps if step not in masks]
    if missing_steps:
        raise ValueError(
            f"Missing requested steps {missing_steps}; available steps: {sorted(masks)}"
        )

    common_layers = set(masks[steps[0]])
    for step in steps[1:]:
        common_layers.intersection_update(masks[step])
    layers = sorted(common_layers)
    if not layers:
        raise ValueError(f"Steps {steps} have no layers in common")

    first_masks = [load_head(masks[step][layers[0]], args.head) for step in steps]
    first_shape = first_masks[0].shape
    for step, mask in zip(steps, first_masks):
        if mask.shape != first_shape:
            raise ValueError(
                f"Step {step}, layer {layers[0]} has shape {mask.shape}, expected {first_shape}"
            )

    panel_width = 5.0
    fig, axes = plt.subplots(
        1,
        len(steps),
        figsize=(panel_width * len(steps), 5.6),
        dpi=120,
        squeeze=False,
    )
    axes = axes[0]
    cmap = ListedColormap(["#f7fbff", "#08306b"])
    images = []
    q_blocks, k_blocks = first_shape
    x_ticks = [0, k_blocks // 2, k_blocks - 1]
    y_ticks = [0, q_blocks // 2, q_blocks - 1]

    for ax, step, mask in zip(axes, steps, first_masks):
        image = ax.imshow(
            mask,
            cmap=cmap,
            vmin=0,
            vmax=1,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )
        images.append(image)
        ax.set_title(f"Step {step}  |  selected {mask.mean():.2%}")
        ax.set_xlabel("Key block")
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    axes[0].set_ylabel("Query block")
    for ax in axes[1:]:
        ax.set_ylabel("")

    title = fig.suptitle("")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame_idx, layer in enumerate(layers):
            for image, ax, step in zip(images, axes, steps):
                mask = load_head(masks[step][layer], args.head)
                if mask.shape != first_shape:
                    raise ValueError(
                        f"Step {step}, layer {layer} has shape {mask.shape}, "
                        f"expected {first_shape}"
                    )
                image.set_data(mask)
                ax.set_title(f"Step {step}  |  selected {mask.mean():.2%}")

            title.set_text(
                f"DFSAttn top-k mask  |  Head {args.head}  |  "
                f"Layer {layer} ({frame_idx + 1}/{len(layers)})"
            )
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)

    duration = len(layers) / args.fps
    print(
        f"Saved {args.output}: {len(layers)} frames, {args.fps:g} fps, "
        f"{duration:.1f}s, head {args.head}, steps {steps}, layers {layers[0]}-{layers[-1]}"
    )


if __name__ == "__main__":
    main()
