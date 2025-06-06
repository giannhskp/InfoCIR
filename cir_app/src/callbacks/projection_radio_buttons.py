from dash import callback, Output, Input
from src.widgets import scatterplot

@callback(
    Output('scatterplot', 'figure', allow_duplicate=True),
    Input('projection-radio-buttons', 'value'),
    prevent_initial_call=True,
)
def projection_radio_is_clicked(radio_button_value):
    """Handle projection radio button changes"""
    print('Radio button is clicked')
    return scatterplot.create_scatterplot_figure(radio_button_value) 