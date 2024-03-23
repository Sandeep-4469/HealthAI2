from functools import partial
import gradio as gr
from bone import bone_main
import os
from glob import glob
import numpy as np
with gr.Blocks(title="Medical Image Segmentation") as demo:
    gr.Markdown("""<h1><center>    Medical Image Segmentation with UW-Madison GI Tract Dataset</center></h1>""")
    with gr.Row():
        img_input = gr.Image(type="pil", height=360, width=360, label="Input image")
        img_output = gr.Image(label="Predictions", height=360, width=360)
    section_btn = gr.Button("Generate Predictions")
    section_btn.click(bone_main, img_input, img_output)
demo.launch()