from dash import Input, Output, callback, State, dash
from src.widgets import scatterplot
import copy

@callback(
    Output('selected-histogram-class', 'data'),
    Input('histogram', 'clickData'),
    prevent_initial_call=True,
)
def histogram_is_clicked(histogram_click):
    """Handle histogram clicks to highlight class on scatterplot"""
    if histogram_click is None:
        return None

    print('Histogram is clicked')
    # Extract the full class name from customdata instead of the truncated x value
    # customdata[0] contains the full class name, while x contains the truncated display name
    clicked_class = histogram_click['points'][0]['customdata'][0]
    print(f'Clicked class: {clicked_class}')
    # Scatterplot highlighting handled by unified controller
    return clicked_class 