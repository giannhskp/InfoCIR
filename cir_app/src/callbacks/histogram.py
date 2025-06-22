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
    clicked_class = histogram_click['points'][0]['x']
    # Scatterplot highlighting handled by unified controller
    return clicked_class 