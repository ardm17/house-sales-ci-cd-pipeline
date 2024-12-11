import gradio as gr
import skops.io as sio

# Attempt to load the moddel, catching the TypeError
try:
    model = sio.load("Model/house_rf_model.skops", trusted=[])
except TypeError as e:
    # If there are untrusted types, retrieve them from the error message
    untrusted_types = sio.get_untrusted_types()
    model = sio.load("Model/house_rf_model.skops", trusted=untrusted_types)

# Define the prediction function
def predict_price(sqft_living, bedrooms, bathrooms, floors, condition, grade, waterfront, yr_built):

    features = [[sqft_living, bedrooms, bathrooms, floors, condition, grade, waterfront, yr_built]]
    prediction = model.predict(features)[0]
    return f"Predicted House Price: ${prediction:.2f}"

# Define the Gradio interface using the updated API
with gr.Blocks() as demo:
    gr.Markdown("# House Price Predictor")

    sqft_living = gr.Slider(500, 5000, label="Square Foot Living")
    bedrooms = gr.Slider(1, 10, label="Bedrooms")
    bathrooms = gr.Slider(1, 5, label="Bathrooms")
    floors = gr.Slider(1, 4, label="Floors")
    condition = gr.Slider(1, 5, label="Condition")
    grade = gr.Slider(1, 13, label="Grade")
    waterfront = gr.Checkbox(label="Waterfront")
    yr_built = gr.Slider(1900, 2021, label="Year Built")

    output = gr.Textbox(label="Predicted Price")

    predict_button = gr.Button("Predict Price")
    predict_button.click(
        predict_price,
        inputs=[sqft_living, bedrooms, bathrooms, floors, condition, grade, waterfront, yr_built],
        outputs=output
    )

demo.launch()