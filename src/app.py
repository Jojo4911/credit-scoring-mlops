import gradio as gr
import random

def minimal_endpoint(income: float, age: int):
    if age > 0:
        if income >= 0:
            ratio = income / age
        else:
            raise gr.Error("Veuillez entrer un revenu supérieur ou égal à 0")
    else:
        raise gr.Error("Veuillez entrer un âge supérieur à 0")

    score = random.random()

    return ratio, score

revenu = gr.Number(label="Veuillez entrer votre revenu annuel")
age = gr.Number(label="Veuillez entrer votre âge")

demo = gr.Interface(
    fn=minimal_endpoint,
    inputs=[revenu, age],
    outputs=[gr.Textbox(label="Ratio revenu / âge :"), gr.Number(label="Score :")]
)

demo.launch()