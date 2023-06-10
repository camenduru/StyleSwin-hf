#!/usr/bin/env python

from __future__ import annotations

import gradio as gr
import numpy as np

from model import Model

DESCRIPTION = '# [StyleSwin](https://github.com/microsoft/StyleSwin)'


def get_sample_image_url(name: str) -> str:
    sample_image_dir = 'https://huggingface.co/spaces/hysts/StyleSwin/resolve/main/samples'
    return f'{sample_image_dir}/{name}.jpg'


def get_sample_image_markdown(name: str) -> str:
    url = get_sample_image_url(name)
    if name == 'celeba-hq':
        size = 1024
    elif name == 'ffhq':
        size = 1024
    elif name == 'lsun-church':
        size = 256
    else:
        raise ValueError
    seed = '0-99'
    return f'''
    - size: {size}x{size}
    - seed: {seed}
    ![sample images]({url})'''


model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('App'):
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(model.MODEL_NAMES,
                                             value=model.MODEL_NAMES[3],
                                             label='Model')
                    seed = gr.Slider(0,
                                     np.iinfo(np.uint32).max,
                                     step=1,
                                     value=0,
                                     label='Seed')
                    run_button = gr.Button('Run')
                with gr.Column():
                    result = gr.Image(label='Result', elem_id='result')

        with gr.TabItem('Sample Images'):
            with gr.Row():
                model_name2 = gr.Dropdown([
                    'celeba-hq',
                    'ffhq',
                    'lsun-church',
                ],
                                          value='celeba-hq',
                                          label='Model')
            with gr.Row():
                text = get_sample_image_markdown(model_name2.value)
                sample_images = gr.Markdown(text)

    run_button.click(fn=model.set_model_and_generate_image,
                     inputs=[model_name, seed],
                     outputs=result,
                     api_name='run')
    model_name2.change(fn=get_sample_image_markdown,
                       inputs=model_name2,
                       outputs=sample_images)

demo.queue(max_size=15).launch()
