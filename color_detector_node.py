"""
@author: christian-byrne
@title: Img2Color Node - Detect and describe color palettes in images
@nickname: img2color
@description:
"""

import torch
import webcolors
from colornamer import get_color_from_rgb # type: ignore
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from typing import Tuple, List, Dict, Any, Optional


class Img2ColorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "num_colors": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Number of colors to detect",
                    },
                ),
                "get_complementary": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_off": "Get Original Colors",
                        "label_on": "Get Complementary Colors",
                        "tooltip": "Get the complementary colors of the detected palette",
                    },
                ),
                "k_means_algorithm": (
                    ["lloyd", "elkan", "auto", "full"],
                    {
                        "default": "lloyd",
                    },
                ),
                "accuracy": (
                    "INT",
                    {
                        "default": 60,
                        "display": "slider",
                        "min": 1,
                        "max": 100,
                        "tooltip": "Adjusts accuracy by changing number of iterations of the K-means algorithm",
                    },
                ),
                "exclude_colors": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated list of colors to exclude from the output",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "output_text": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "plain_english_colors",
        "rgb_colors",
        "hex_colors",
        "xkcd_colors",
        "design_colors",
        "common_colors",
        "color_types",
        "color_families",
    )
    OUTPUT_TOOLTIPS = (
        "Plain English Colors correspond to the closest named color in the CSS3, CSS2, CSS21, and HTML4 color dictionaries.",
        'RGB Colors are in the format "rgb(255, 0, 255)"',
        "Hex Colors are in the format #RRGGBB",
        "XKCD Color is the finest level of granularity, and corresponds to the colors in the XKCD color survey. There are about 950 colors in this space.",
        "Design Color is the next coarsest level. There are about 250 Design Colors",
        "Common Color is the next coarsest level. There are about 120 Common Colors. This is probably the most useful level for most purposes.",
        "Color Type is another dimension that tells, roughly, how light, dark or saturated a color is. There are 11 color types.",
        "Color Family is even coarser, and has 26 families. These are all primary, secondary, or tertiary colors, or corresponding values for neutrals.",
    )
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "img2txt"

    def main(
        self,
        input_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        num_colors: int = 5,
        k_means_algorithm: str = "lloyd",
        accuracy: int = 80,
        get_complementary: bool = False,
        exclude_colors: str = "",
        output_text: str = "",
        unique_id: Optional[str] = None,
        extra_pnginfo: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ...]:
        if exclude_colors.strip():
            self.exclude: List[str] = exclude_colors.strip().split(",")
            self.exclude = [color.strip().lower() for color in self.exclude]
        else:
            self.exclude = []
        num_colors = max(1, num_colors)
        self.num_iterations = int(512 * (accuracy / 100))
        self.algorithm = k_means_algorithm
        self.webcolor_dict: Dict[str, str] = {}
        specs = [webcolors.CSS2, webcolors.CSS21, webcolors.CSS3, webcolors.HTML4]
        for spec in specs:
            try:
                for name in webcolors.names(spec):
                    normalized_name = name.lower()
                    hex_val = webcolors.name_to_hex(normalized_name, spec=spec)
                    if hex_val not in self.webcolor_dict or len(normalized_name) < len(self.webcolor_dict[hex_val]):
                         self.webcolor_dict[hex_val] = normalized_name
            except ValueError:
                 print(f"Warning: Could not process spec '{spec}' in webcolors.") # noqa: T201
                 continue
            except AttributeError:
                 print(f"Warning: webcolors.names function not found for spec '{spec}'. Skipping.") # noqa: T201
                 continue

        seed = self.try_get_seed(extra_pnginfo) if extra_pnginfo else None
        original_colors = self.interrogate_colors(
            input_image, num_colors, seed
        )
        rgb = self.ndarrays_to_rgb(original_colors)
        if get_complementary:
            rgb = self.rgb_to_complementary(rgb)

        plain_english_colors = [self.get_webcolor_name(color) for color in rgb]
        rgb_colors = [f"rgb{color}" for color in rgb]
        hex_colors = [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in rgb]

        colornamer_names: List[Dict[str, str]] = self.get_colornamer_names(rgb)
        xkcd_colors = [color.get("xkcd_color", "unknown") for color in colornamer_names]
        design_colors = [color.get("design_color", "unknown") for color in colornamer_names]
        common_colors = [color.get("common_color", "unknown") for color in colornamer_names]
        color_types = [color.get("color_type", "unknown") for color in colornamer_names]
        color_families = [color.get("color_family", "unknown") for color in colornamer_names]

        return (
            self.join_and_exclude(plain_english_colors),
            self.join_and_exclude(rgb_colors),
            self.join_and_exclude(hex_colors),
            self.join_and_exclude(xkcd_colors),
            self.join_and_exclude(design_colors),
            self.join_and_exclude(common_colors),
            self.join_and_exclude(color_types),
            self.join_and_exclude(color_families),
        )

    def join_and_exclude(self, colors: List[str]) -> str:
        return ", ".join(
            [str(color) for color in colors if color.lower() not in getattr(self, "exclude", [])]
        )

    def get_colornamer_names(self, colors: List[Tuple[int, int, int]]) -> List[Dict[str, str]]:
        results = []
        for color in colors:
            try:
                color_info = get_color_from_rgb(list(float(c) for c in color))
                results.append(color_info if isinstance(color_info, dict) else {"error": "unexpected format"})
            except Exception as e:
                print(f"Error calling colornamer for {color}: {e}") # noqa: T201
                results.append({"error": str(e)}) # Append error dict
        return results

    def rgb_to_complementary(
        self, colors: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        return [(255 - color[0], 255 - color[1], 255 - color[2]) for color in colors]

    def ndarrays_to_rgb(self, colors: NDArray[np.float_]) -> List[Tuple[int, int, int]]:
        if colors.ndim == 1:
             return [(int(round(colors[0])), int(round(colors[1])), int(round(colors[2])))]
        return [(int(round(color[0])), int(round(color[1])), int(round(color[2]))) for color in colors]

    def interrogate_colors(
        self, image: torch.Tensor, num_colors: int, seed: Optional[int] = None
    ) -> NDArray[np.float_]:
        pixels: NDArray[np.float_] = image.view(-1, image.shape[-1]).to(dtype=torch.float32).numpy()
        num_unique_pixels = len(np.unique(pixels, axis=0))
        actual_num_colors = min(num_colors, num_unique_pixels)
        if actual_num_colors < 1:
            if pixels.shape[0] > 0:
                return (pixels[0:1] * 255.0).astype(np.float_)
            else:
                return np.array([[0.0, 0.0, 0.0]], dtype=np.float_)

        kmeans = KMeans(
                n_clusters=actual_num_colors,
                algorithm=self.algorithm, # type: ignore
                max_iter=self.num_iterations,
                random_state=seed,
                n_init=10
            )
        if actual_num_colors > 0 and pixels.shape[0] > 0:
            kmeans.fit(pixels)
            colors: NDArray[np.float_] = kmeans.cluster_centers_ * 255.0
        else:
             colors = np.array([[0.0, 0.0, 0.0]], dtype=np.float_)

        return colors.astype(np.float_)

    def get_webcolor_name(self, rgb: Tuple[int, int, int]) -> str:
        closest_match: str = "unknown"
        min_distance = float("inf")

        if not hasattr(self, "webcolor_dict") or not self.webcolor_dict:
             print("Warning: webcolor_dict not initialized or empty.") # noqa: T201
             return closest_match # Should not happen if main() runs first

        for hex_val, name in self.webcolor_dict.items():
            try:
                web_rgb = webcolors.hex_to_rgb(hex_val)
                distance = sum(abs(a - b) for a, b in zip(rgb, web_rgb))
                if distance < min_distance:
                    min_distance = distance
                    closest_match = name
            except ValueError:
                 print(f"Warning: Invalid hex '{hex_val}' found in webcolor dictionary.") # noqa: T201
                 continue

        return closest_match

    def try_get_seed(self, extra_pnginfo: Dict[str, Any]) -> Optional[int]:
        try:
            workflow = extra_pnginfo.get("workflow", {})
            nodes = workflow.get("nodes", [])
            for node in nodes:
                 node_type = node.get("type")
                 widgets_values = node.get("widgets_values")
                 if node_type and ("KSampler" in node_type) and isinstance(widgets_values, list) and len(widgets_values) > 0:
                     seed_value = widgets_values[0]
                     if isinstance(seed_value, (int, float)):
                         seed = int(seed_value)
                         if seed >= 0:
                            return seed
                         else:
                             print(f"Warning: Negative seed {seed} encountered.") # noqa: T201
                             return None # Kmeans random_state needs non-negative

        except Exception as e:
             print(f"Error parsing seed from extra_pnginfo: {e}") # noqa: T201
             pass # Keep original behavior of returning None on error
        return None
