from dash import callback, Output, Input, State
from src.widgets import scatterplot
from src.Dataset import Dataset
from src import config
import plotly.graph_objects as go

@callback(
    Output('scatterplot', 'id'),
    Input('projection-radio-buttons', 'value'),
    prevent_initial_call=True,
)
def projection_radio_buttons_are_clicked(projection_selected):
    """Handle projection radio button changes"""
    print(f'Projection changed to: {projection_selected}')
    # Scatterplot figure creation handled by unified controller
    return 'scatterplot'

@callback(
    [Output('selected-image-data', 'data', allow_duplicate=True),
     Output('selected-gallery-image-ids', 'data', allow_duplicate=True)],
    Input('projection-radio-buttons', 'value'),
    [State('cir-toggle-state', 'data'),
     State('cir-search-data', 'data')],
    prevent_initial_call=True,
)
def projection_radio_is_clicked(radio_button_value, cir_toggle_state, search_data):
    """Handle projection radio button clicks"""
    print('Projection radio button is clicked')
    # Scatterplot figure creation handled by unified controller
    return None, [] 