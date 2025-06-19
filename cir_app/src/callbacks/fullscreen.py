from dash import callback, Output, Input, State, html, no_update

# Callback to toggle fullscreen mode for the "Query Results" card
@callback(
    [Output('cir-results-card', 'style'),
     Output('cir-results-fullscreen', 'data'),
     Output('cir-results-expand-btn', 'children'),
     Output('cir-results-expand-btn', 'color')],
    Input('cir-results-expand-btn', 'n_clicks'),
    State('cir-results-fullscreen', 'data'),
    prevent_initial_call=True
)
def toggle_query_results_fullscreen(n_clicks, is_fullscreen):
    """Toggle fullscreen view for the Query Results card.

    When the expand button is clicked we switch the card between its normal
    position inside the left column and a fixed, viewport-filling overlay. The
    same button (now showing a "compress" icon) can be pressed again to restore
    the original layout.
    """

    # Dash provides None on first load; normalise to False.
    is_fullscreen = bool(is_fullscreen)

    # Toggle state
    new_state = not is_fullscreen

    if new_state:
        # Enter fullscreen – fixed overlay covering the whole viewport.
        style = {
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'width': '100vw',
            'height': '100vh',
            'zIndex': 2000,  # sufficiently above other elements
            'backgroundColor': 'white',
            'overflow': 'auto',
            'padding': '1rem',
        }
        btn_icon = html.I(className='fas fa-compress')
        btn_color = 'secondary'
    else:
        # Exit fullscreen – revert to original inline card style.
        style = {'flex': '1 1 25%', 'overflow': 'auto'}
        btn_icon = html.I(className='fas fa-expand')
        btn_color = 'outline-secondary'

    return style, new_state, btn_icon, btn_color

# Callback to toggle fullscreen mode for the Histogram/Wordcloud card
@callback(
    [Output('hist-wh-card', 'style'),
     Output('hist-wh-fullscreen', 'data'),
     Output('wh-expand-btn', 'children'),
     Output('wh-expand-btn', 'color')],
    Input('wh-expand-btn', 'n_clicks'),
    State('hist-wh-fullscreen', 'data'),
    prevent_initial_call=True
)
def toggle_hist_wordcloud_fullscreen(n_clicks, is_fullscreen):
    """Toggle fullscreen view for the Histogram/Wordcloud card."""
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen

    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh', 'zIndex': 2000,
            'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        btn_icon = html.I(className='fas fa-compress')
        btn_color = 'secondary'
    else:
        style = {'flex': '1 1 25%', 'overflow': 'auto'}
        btn_icon = html.I(className='fas fa-expand')
        btn_color = 'outline-secondary'

    return style, new_state, btn_icon, btn_color

# Callback to toggle fullscreen mode for the Prompt Enhancement card
@callback(
    [Output('prompt-enh-card', 'style'),
     Output('prompt-enh-fullscreen', 'data'),
     Output('prompt-enh-expand-btn', 'children'),
     Output('prompt-enh-expand-btn', 'color')],
    Input('prompt-enh-expand-btn', 'n_clicks'),
    State('prompt-enh-fullscreen', 'data'),
    prevent_initial_call=True
)
def toggle_prompt_enh_fullscreen(n_clicks, is_fullscreen):
    """Toggle fullscreen for the Prompt Enhancement panel."""
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh', 'zIndex': 2000,
            'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        icon = html.I(className='fas fa-compress')
        color = 'secondary'
    else:
        style = {'flex': '1 1 25%', 'overflow': 'auto'}
        icon = html.I(className='fas fa-expand')
        color = 'outline-secondary'
    return style, new_state, icon, color

# Callback to toggle fullscreen mode for the Rank-Δ card
@callback(
    [Output('rank-delta-card', 'style'),
     Output('rank-delta-fullscreen', 'data'),
     Output('rank-delta-expand-btn', 'children'),
     Output('rank-delta-expand-btn', 'color')],
    Input('rank-delta-expand-btn', 'n_clicks'),
    State('rank-delta-fullscreen', 'data'),
    prevent_initial_call=True
)
def toggle_rank_delta_fullscreen(n_clicks, is_fullscreen):
    """Toggle fullscreen for Rank-Δ panel."""
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh', 'zIndex': 2000,
            'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        icon = html.I(className='fas fa-compress')
        color = 'secondary'
    else:
        style = {'flex': '1 1 25%', 'overflow': 'auto'}
        icon = html.I(className='fas fa-expand')
        color = 'outline-secondary'
    return style, new_state, icon, color

# Callback to toggle fullscreen mode for the Saliency card
@callback(
    [Output('saliency-card', 'style'),
     Output('saliency-fullscreen', 'data'),
     Output('saliency-expand-btn', 'children'),
     Output('saliency-expand-btn', 'color')],
    Input('saliency-expand-btn', 'n_clicks'),
    State('saliency-fullscreen', 'data'),
    prevent_initial_call=True
)
def toggle_saliency_fullscreen(n_clicks, is_fullscreen):
    """Toggle fullscreen for Saliency panel."""
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh', 'zIndex': 2000,
            'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        icon = html.I(className='fas fa-compress')
        color = 'secondary'
    else:
        style = {'flex': '1 1 25%', 'overflow': 'hidden', 'display': 'flex', 'flexDirection': 'column'}
        icon = html.I(className='fas fa-expand')
        color = 'outline-secondary'
    return style, new_state, icon, color

# Callback to toggle fullscreen mode for the Token Attribution card
@callback(
    [Output('token-attr-card', 'style'),
     Output('token-attr-fullscreen', 'data'),
     Output('token-attr-expand-btn', 'children'),
     Output('token-attr-expand-btn', 'color')],
    Input('token-attr-expand-btn', 'n_clicks'),
    State('token-attr-fullscreen', 'data'),
    prevent_initial_call=True
)
def toggle_token_attr_fullscreen(n_clicks, is_fullscreen):
    """Toggle fullscreen for Token Attribution panel."""
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh', 'zIndex': 2000,
            'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        icon = html.I(className='fas fa-compress')
        color = 'secondary'
    else:
        style = {'flex': '1 1 25%', 'overflow': 'auto'}
        icon = html.I(className='fas fa-expand')
        color = 'outline-secondary'
    return style, new_state, icon, color

# Callback to toggle fullscreen mode for the CIR Controls card
@callback(
    [Output('cir-controls-card', 'style'),
     Output('cir-controls-fullscreen', 'data'),
     Output('cir-controls-expand-btn', 'children'),
     Output('cir-controls-expand-btn', 'color')],
    Input('cir-controls-expand-btn', 'n_clicks'),
    State('cir-controls-fullscreen', 'data'),
    prevent_initial_call=True
)
def toggle_cir_controls_fullscreen(n_clicks, is_fullscreen):
    """Toggle fullscreen for the CIR Controls panel."""
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh', 'zIndex': 2000,
            'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        icon = html.I(className='fas fa-compress')
        color = 'secondary'
    else:
        style = {'flex': '1 1 25%', 'overflow': 'auto'}
        icon = html.I(className='fas fa-expand')
        color = 'outline-secondary'
    return style, new_state, icon, color 