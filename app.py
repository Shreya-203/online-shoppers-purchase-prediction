import gradio as gr
import pandas as pd
import joblib

# Load the trained model and preprocessor
best_model = joblib.load("gradient_boosting_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


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

    # Apply the saved preprocessing pipeline
    processed_data = preprocessor.transform(input_data)

    # Predict purchase outcome
    prediction = best_model.predict(processed_data)[0]

    # Calculate probabilities
    probabilities = best_model.predict_proba(processed_data)[0]
    classes = list(best_model.classes_)

    no_purchase_probability = probabilities[classes.index(False)]
    purchase_probability = probabilities[classes.index(True)]

    if prediction == True:
        result = "Purchase Likely"
    else:
        result = "No Purchase Likely"

    probability_text = (
        f"No Purchase: {no_purchase_probability:.2%} | "
        f"Purchase: {purchase_probability:.2%}"
    )

    return result, probability_text


# Create the web interface
demo = gr.Interface(
    fn=predict_purchase,
    inputs=[
        gr.Number(label="Administrative Pages", value=0, precision=0),
        gr.Number(label="Administrative Duration", value=0),
        gr.Number(label="Informational Pages", value=0, precision=0),
        gr.Number(label="Informational Duration", value=0),
        gr.Number(label="Product Related Pages", value=5, precision=0),
        gr.Number(label="Product Related Duration", value=100),
        gr.Number(label="Bounce Rate", value=0.02),
        gr.Number(label="Exit Rate", value=0.04),
        gr.Number(label="Page Values", value=0),
        gr.Number(label="Special Day", value=0),
        gr.Dropdown(
            choices=[
                "Feb", "Mar", "May", "June", "Jul",
                "Aug", "Sep", "Oct", "Nov", "Dec"
            ],
            label="Month",
            value="May"
        ),
        gr.Dropdown(
            choices=list(range(1, 9)),
            label="Operating System",
            value=2
        ),
        gr.Dropdown(
            choices=list(range(1, 14)),
            label="Browser",
            value=2
        ),
        gr.Dropdown(
            choices=list(range(1, 10)),
            label="Region",
            value=1
        ),
        gr.Dropdown(
            choices=list(range(1, 21)),
            label="Traffic Type",
            value=2
        ),
        gr.Dropdown(
            choices=[
                "Returning_Visitor",
                "New_Visitor",
                "Other"
            ],
            label="Visitor Type",
            value="Returning_Visitor"
        ),
        gr.Checkbox(label="Weekend", value=False)
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Purchase Probability")
    ],
    title="Online Shoppers Purchase Intention Prediction",
    description=(
        "Enter online shopping session details to estimate "
        "whether a visitor is likely to make a purchase."
    )
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=10000
    )