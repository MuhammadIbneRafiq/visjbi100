import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Initialize the app
app = dash.Dash(__name__)
app.title = "Dummy Dashboard"

# Define app layout
app.layout = html.Div([
    html.H1("Dummy Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.H2("Section 1: Controls", style={"marginTop": "20px"}),

        html.Label("Slider 1:"),
        dcc.Slider(id="slider-1", min=0, max=100, step=1, value=50),
        html.Div(id="slider-1-output", style={"marginBottom": "20px"}),

        html.Label("Slider 2:"),
        dcc.Slider(id="slider-2", min=0, max=50, step=5, value=25),
        html.Div(id="slider-2-output", style={"marginBottom": "20px"}),
    ]),

    html.Div([
        html.H2("Section 2: Dummy Charts", style={"marginTop": "40px"}),
        dcc.Graph(
            id="dummy-bar-chart",
            figure={
                "data": [
                    {"x": ["A", "B", "C"], "y": [10, 15, 7], "type": "bar", "name": "Sample Data"}
                ],
                "layout": {"title": "Dummy Bar Chart"}
            }
        ),

        dcc.Graph(
            id="dummy-line-chart",
            figure={
                "data": [
                    {"x": [0, 1, 2, 3, 4], "y": [5, 10, 15, 10, 5], "type": "line", "name": "Line Data"}
                ],
                "layout": {"title": "Dummy Line Chart"}
            }
        ),
    ]),

    html.Div([
        html.H2("Section 3: Summary", style={"marginTop": "40px"}),
        html.P("This is a dummy dashboard for testing purposes. Add your production logic here!", style={"fontSize": "16px"}),
    ], style={"textAlign": "center"})
])

# Callbacks for slider outputs
@app.callback(
    Output("slider-1-output", "children"),
    Input("slider-1", "value")
)
def update_slider_1_output(value):
    return f"Slider 1 Value: {value}"

@app.callback(
    Output("slider-2-output", "children"),
    Input("slider-2", "value")
)
def update_slider_2_output(value):
    return f"Slider 2 Value: {value}"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
