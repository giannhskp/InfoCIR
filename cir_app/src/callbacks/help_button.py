from dash import Output, callback, Input, State
from dash import callback_context

@callback(
    Output("help-popup", "is_open"),
    [Input("help-button", "n_clicks")],
    [State("help-popup", "is_open")],
    prevent_initial_call=True,
)
def toggle_help_popup(help_clicks, is_open):
    """Handle help button clicks to toggle the popup"""
    # Get the triggering component
    ctx = callback_context
    if not ctx.triggered:
        return False
    
    # Get the button that was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Help button toggles the popup state
    if button_id == "help-button":
        return not is_open if is_open is not None else True
    
    # Fallback: return current state
    return is_open if is_open is not None else False 