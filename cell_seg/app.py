"""Gradio UI entry point."""
import gradio as gr


def build_ui() -> gr.Blocks:
    raise NotImplementedError


if __name__ == "__main__":
    build_ui().launch()
