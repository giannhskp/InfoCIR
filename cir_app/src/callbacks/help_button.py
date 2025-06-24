from dash import Output, callback, Input, State
from dash import callback_context

@callback(
    Output("help-popup", "is_open"),
    [Input("help-button", "n_clicks"),
     Input("close-help", "n_clicks")],
    [State("help-popup", "is_open")],
    prevent_initial_call=True,
)
def toggle_help_popup(help_clicks, close_clicks, is_open):
    """Handle help button clicks to toggle the popup"""
    # Get the triggering component
    ctx = callback_context
    if not ctx.triggered:
        return False
    
    # Get the button that was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Simple logic: help button opens, close button closes
    if button_id == "help-button":
        return True
    elif button_id == "close-help":
        return False
    
    # Fallback: return current state
    return is_open if is_open is not None else False 