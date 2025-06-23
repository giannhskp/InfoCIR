from dash import Input, Output, callback, State
import plotly.graph_objects as go
from src.widgets import scatterplot

@callback(
    [Output('selected-gallery-image-ids', 'data', allow_duplicate=True),
     Output('cir-toggle-button', 'children', allow_duplicate=True),
     Output('cir-toggle-button', 'color', allow_duplicate=True),
     Output('cir-toggle-state', 'data', allow_duplicate=True),
     Output('prompt-selection', 'value', allow_duplicate=True),
     Output('histogram', 'figure', allow_duplicate=True),
     Output('selected-histogram-class', 'data', allow_duplicate=True),
     Output('selected-scatterplot-class', 'data', allow_duplicate=True)],
    [State('projection-radio-buttons', 'value'),
     State('scatterplot', 'figure'),
     State('cir-toggle-state', 'data')],
    Input('deselect-button', 'n_clicks'),
    prevent_initial_call=True,
)
def deselect_button_is_pressed(projection_selected, scatterplot_fig, cir_toggle_state, _):
    """Handle deselect button clicks"""
    print('Deselect button is clicked')

    import copy

    # ------------------------------------------------------------------
    # 1. Update the *Visualize CIR results* toggle button.
    #    Requirement: if visualisation was ON before clicking "Deselect",
    #    auto-enable it again afterwards (i.e. keep it ON).  So we *retain*
    #    the previous ON state instead of turning it OFF.
    # ------------------------------------------------------------------
    if cir_toggle_state:
        # Visualisation was ON – keep it ON
        cir_button_text = 'Hide CIR results'
        cir_button_color = 'warning'
        new_cir_toggle_state = True
    else:
        # Visualisation was OFF – keep it OFF (unchanged behaviour)
        cir_button_text = 'Visualize CIR results'
        cir_button_color = 'success'
        new_cir_toggle_state = False

    # ------------------------------------------------------------------
    # 2. Build a blank histogram (since everything is deselected).
    # ------------------------------------------------------------------
    from src.widgets import histogram as histogram_widget
    histogram_fig = histogram_widget.draw_histogram(None)

    return [], cir_button_text, cir_button_color, new_cir_toggle_state, -1, histogram_fig, None, None 