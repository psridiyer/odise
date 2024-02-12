
import os
# we have made some changes like changing np.bool to bool
# changed path for modules
# need to install from our own repo. not from NVlabs 
# following will not work
#os.system("pip install git+https://github.com/NVlabs/ODISE.git")
#os.system("pip freeze")

import itertools
import json
from contextlib import ExitStack
import gradio as gr
import numpy as np
import matplotlib.colors as mplc
import torch
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from PIL import Image
from torch.cuda.amp import autocast

from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer as _Visualizer, random_color

from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.modeling.wrapper import OpenPanopticInference

setup_logger()
logger = setup_logger(name="odise")

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES))
)

class Visualizer(_Visualizer):

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.clip(color, 0, 1).tolist()
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

class VisualizationDemo(object):
    def __init__(self, model, metadata, aug, instance_mode=ColorMode.IMAGE):
        """
        Args:
            model (nn.Module):
            metadata (MetadataCatalog): image metadata.
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def predict(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        logger.info("forwarding")
        with autocast():
            predictions = self.model([inputs])[0]
        logger.info("done")
        return predictions

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predict(image)
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


cfg = model_zoo.get_config("Panoptic/odise_label_coco_50e.py", trained=True)

cfg.model.overlap_threshold = 0
cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"
seed_all_rng(42)

dataset_cfg = cfg.dataloader.test
wrapper_cfg = cfg.dataloader.wrapper

aug = instantiate(dataset_cfg.mapper).augmentations

model = instantiate_odise(cfg.model)
model.to(torch.float16)
model.to(cfg.train.device)
ODISECheckpointer(model).load(cfg.train.init_checkpoint)


title = "Spector AI"
description = """

Demo of Spector AI models for segmenting objects in an image and also identify and
recognize such objects (provide a label).

Please start with one of the examples. You can see the indentification of objects in the image.

You may upload any image. Lables for objects in an image may be wrong. You can provide additional
vocabulary to help the system. The extra vocabulary is meant to guide the system. Enter such guidance
in the filed called Extra Vocabulary (words spearated by comma).
"""  # noqa

article = """
<p style='text-align: center'><a href='https://arxiv.org/abs/2303.04803' target='_blank'>Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models</a> | <a href='https://github.com/NVlab/ODISE' target='_blank'>Github Repo</a></p>
"""  # noqa

examples = [
    [
        "demo/examples/coco.jpg",
        "black pickup truck, pickup truck; blue sky, sky",
        ["COCO (133 categories)", "ADE (150 categories)", "LVIS (1203 categories)"],
    ],
    [
        "demo/examples/ade.jpg",
        "luggage, suitcase, baggage;handbag",
        ["ADE (150 categories)"],
    ],
    [
        "demo/examples/ego4d.jpg",
        "faucet, tap; kitchen paper, paper towels",
        ["COCO (133 categories)"],
    ],
]


def build_demo_classes_and_metadata(vocab, label_list):
    extra_classes = []

    if vocab:
        for words in vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if any("COCO" in label for label in label_list):
        demo_thing_classes += COCO_THING_CLASSES
        demo_stuff_classes += COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors += COCO_STUFF_COLORS
    if any("ADE" in label for label in label_list):
        demo_thing_classes += ADE_THING_CLASSES
        demo_stuff_classes += ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if any("LVIS" in label for label in label_list):
        demo_thing_classes += LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS

    MetadataCatalog.pop("odise_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_classes, demo_metadata


def inference(image_path, vocab, label_list):

    logger.info("building class names")
    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model,
            labels=demo_classes,
            metadata=demo_metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True,
        )
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = VisualizationDemo(inference_model, demo_metadata, aug)
        img = utils.read_image(image_path, format="RGB")
        _, visualized_output = demo.run_on_image(img)
        return Image.fromarray(visualized_output.get_image())

with gr.Blocks(title=title) as demo:
    
    gr.Image(value="./sp_img/spector_vert.png")
    gr.Markdown(description)
    input_components = []
    output_components = []

    with gr.Row():
        output_image_gr = gr.Image(label="Panoptic Segmentation", type="pil")
        output_components.append(output_image_gr)

    with gr.Row():
        with gr.Column(scale=3, variant="panel") as input_component_column:
            input_image_gr = gr.Image(type="filepath")
            extra_vocab_gr = gr.Textbox( label="Extra Vocabulary")
            category_list_gr = gr.CheckboxGroup(
                choices=["COCO (133 categories)", "ADE (150 categories)", "LVIS (1203 categories)"],
                label="Category to use",
            )
            input_components.extend([input_image_gr, extra_vocab_gr, category_list_gr])

        with gr.Column(scale=2):
            examples_handler = gr.Examples(
                examples=examples,
                inputs=[c for c in input_components if not isinstance(c, gr.State)],
                outputs=[c for c in output_components if not isinstance(c, gr.State)],
                fn=inference,
                #cache_examples=torch.cuda.is_available(),
                examples_per_page=5,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")

    gr.Markdown(article)

    submit_btn.click(
        inference,
        input_components,
        output_components,
        api_name="predict",
        scroll_to_output=True,
    )

    clear_btn.click(
        None,
        [],
        (input_components + output_components + [input_component_column]),
       
    )

demo.launch(server_name="0.0.0.0", server_port=8080)
#demo.launch(share=True)
