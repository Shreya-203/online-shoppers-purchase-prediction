import gradio as gr
import pandas as pd
import joblib

# Load the trained model and preprocessing pipeline
best_model = joblib.load("gradient_boosting_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


# Predict online shopping purchase intention
def predict_purchase(
    Administrative,
    Administrative_Duration,
    Informational,
    Informational_Duration,
    ProductRelated,
    ProductRelated_Duration,
    BounceRates,
    ExitRates,
    PageValues,
    SpecialDay,
    Month,
    OperatingSystems,
    Browser,
    Region,
    TrafficType,
    VisitorType,
    Weekend
):
    input_data = pd.DataFrame([{
        "Administrative": Administrative,
        "Administrative_Duration": Administrative_Duration,
        "Informational": Informational,
        "Informational_Duration": Informational_Duration,
        "ProductRelated": ProductRelated,
        "ProductRelated_Duration": ProductRelated_Duration,
        "BounceRates": BounceRates,
        "ExitRates": ExitRates,
        "PageValues": PageValues,
        "SpecialDay": SpecialDay,
        "Month": Month,
        "OperatingSystems": OperatingSystems,
        "Browser": Browser,
        "Region": Region,
        "TrafficType": TrafficType,
        "VisitorType": VisitorType,
        "Weekend": Weekend
    }])

    processed_data = preprocessor.transform(input_data)

    prediction = best_model.predict(processed_data)[0]
    probabilities = best_model.predict_proba(processed_data)[0]

    classes = list(best_model.classes_)
    no_purchase_probability = probabilities[classes.index(False)]
    purchase_probability = probabilities[classes.index(True)]

    if prediction == True:
        result = "## ✅ Purchase Likely"
        explanation = (
            "The model estimates that this session is likely "
            "to result in a purchase."
        )
    else:
        result = "## ℹ️ No Purchase Likely"
        explanation = (
            "The model estimates that this session is less likely "
            "to result in a purchase."
        )

    probability_output = f"""
### 📊 Prediction Probabilities

| Outcome | Estimated Probability |
|---|---:|
| Purchase | **{purchase_probability:.2%}** |
| No Purchase | **{no_purchase_probability:.2%}** |

*These probabilities are model estimates, not guarantees of actual customer behavior.*
"""

    return result, explanation, probability_output


# Reset all input values and prediction results
def reset_form():
    return (
        0,                  # Administrative
        0,                  # Administrative_Duration
        0,                  # Informational
        0,                  # Informational_Duration
        5,                  # ProductRelated
        100,                # ProductRelated_Duration
        0.02,               # BounceRates
        0.04,               # ExitRates
        0,                  # PageValues
        0,                  # SpecialDay
        "May",              # Month
        2,                  # OperatingSystems
        2,                  # Browser
        1,                  # Region
        2,                  # TrafficType
        "Returning_Visitor",# VisitorType
        False,              # Weekend
        "",                 # Prediction result
        "",                 # Explanation
        ""                  # Probability output
    )


# Custom CSS
custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
#predict-button {
    font-weight: bold;
}
"""


# Build the application
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="Online Shopping Purchase Predictor"
) as demo:

    gr.Markdown(
        """
        # 🛍️ Online Shoppers Purchase Intention Predictor

        ### Machine Learning Based Purchase Prediction

        Enter the details of an online shopping session to estimate
        whether the visitor is likely to make a purchase.
        """
    )

    # Browsing and session details
    with gr.Row():

        with gr.Column():
            gr.Markdown("### 📄 Browsing Activity")

            Administrative = gr.Number(
                label="Administrative Pages",
                value=0,
                precision=0
            )

            Administrative_Duration = gr.Number(
                label="Administrative Duration",
                value=0
            )

            Informational = gr.Number(
                label="Informational Pages",
                value=0,
                precision=0
            )

            Informational_Duration = gr.Number(
                label="Informational Duration",
                value=0
            )

            ProductRelated = gr.Number(
                label="Product-Related Pages",
                value=5,
                precision=0
            )

            ProductRelated_Duration = gr.Number(
                label="Product-Related Duration",
                value=100
            )

        with gr.Column():
            gr.Markdown("### 📈 Session Behavior")

            BounceRates = gr.Number(
                label="Bounce Rate",
                value=0.02
            )

            ExitRates = gr.Number(
                label="Exit Rate",
                value=0.04
            )

            PageValues = gr.Number(
                label="Page Values",
                value=0
            )

            SpecialDay = gr.Number(
                label="Special Day",
                value=0
            )

            Month = gr.Dropdown(
                choices=[
                    "Feb", "Mar", "May", "June", "Jul",
                    "Aug", "Sep", "Oct", "Nov", "Dec"
                ],
                label="Month",
                value="May"
            )

    # Visitor information
    with gr.Row():

        with gr.Column():
            gr.Markdown("### 💻 Visitor Information")

            OperatingSystems = gr.Dropdown(
                choices=list(range(1, 9)),
                label="Operating System",
                value=2
            )

            Browser = gr.Dropdown(
                choices=list(range(1, 14)),
                label="Browser",
                value=2
            )

            Region = gr.Dropdown(
                choices=list(range(1, 10)),
                label="Region",
                value=1
            )

        with gr.Column():
            gr.Markdown("### 👤 Visitor Type")

            TrafficType = gr.Dropdown(
                choices=list(range(1, 21)),
                label="Traffic Type",
                value=2
            )

            VisitorType = gr.Dropdown(
                choices=[
                    "Returning_Visitor",
                    "New_Visitor",
                    "Other"
                ],
                label="Visitor Type",
                value="Returning_Visitor"
            )

            Weekend = gr.Checkbox(
                label="Was this session on a weekend?",
                value=False
            )

    # Prediction and reset buttons
    with gr.Row():
        predict_button = gr.Button(
            "🔍 Predict Purchase Intention",
            variant="primary",
            elem_id="predict-button"
        )

        reset_button = gr.Button(
            "🔄 Reset",
            variant="secondary"
        )

    # Prediction results
    gr.Markdown("---")
    gr.Markdown("## Prediction Results")

    prediction_output = gr.Markdown()
    explanation_output = gr.Markdown()
    probability_output = gr.Markdown()

    # Input components
    all_inputs = [
        Administrative,
        Administrative_Duration,
        Informational,
        Informational_Duration,
        ProductRelated,
        ProductRelated_Duration,
        BounceRates,
        ExitRates,
        PageValues,
        SpecialDay,
        Month,
        OperatingSystems,
        Browser,
        Region,
        TrafficType,
        VisitorType,
        Weekend
    ]

    # Connect prediction button
    predict_button.click(
        fn=predict_purchase,
        inputs=all_inputs,
        outputs=[
            prediction_output,
            explanation_output,
            probability_output
        ]
    )

    # Connect reset button
    reset_button.click(
        fn=reset_form,
        inputs=[],
        outputs=all_inputs + [
            prediction_output,
            explanation_output,
            probability_output
        ]
    )

    # Footer
    gr.Markdown(
        """
        ---
        **Note:** This application is intended for educational and
        demonstration purposes. Predictions are estimates based on
        a trained machine learning model.
        """
    )


# Launch the application
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=10000
    )

