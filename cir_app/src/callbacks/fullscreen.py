from dash import callback, Output, Input, State, html, no_update

# Callback to toggle fullscreen mode for the "Query Results" card
@callback(
    [Output('cir-results-card', 'style'),
     Output('cir-results-fullscreen', 'data'),
     Output('cir-results-expand-btn', 'children'),
     Output('cir-results-expand-btn', 'color'),
     Output('cir-results-card-style', 'data')],
    Input('cir-results-expand-btn', 'n_clicks'),
    State('cir-results-fullscreen', 'data'),
    State('cir-results-card', 'style'),
    State('cir-results-card-style', 'data'),
    prevent_initial_call=True
)
def toggle_query_results_fullscreen(n_clicks, is_fullscreen, current_style, cached_style):
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
            'zIndex': 2000, 'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        return style, new_state, html.I(className='fas fa-compress'), 'secondary', current_style
    else:
        restored_style = cached_style or {'flex': '1 1 25%', 'overflow': 'auto'}
        return restored_style, new_state, html.I(className='fas fa-expand fa-xs'), 'outline-secondary', no_update

# Callback to toggle fullscreen mode for the Histogram/Wordcloud card
@callback(
    [Output('hist-wh-card', 'style'),
     Output('hist-wh-fullscreen', 'data'),
     Output('wh-expand-btn', 'children'),
     Output('wh-expand-btn', 'color'),
     Output('hist-wh-card-style', 'data')],
    Input('wh-expand-btn', 'n_clicks'),
    State('hist-wh-fullscreen', 'data'),
    State('hist-wh-card', 'style'),
    State('hist-wh-card-style', 'data'),
    prevent_initial_call=True
)
def toggle_hist_wordcloud_fullscreen(n_clicks, is_fullscreen, current_style, cached_style):
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
            'zIndex': 2000, 'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        return style, new_state, html.I(className='fas fa-compress'), 'secondary', current_style
    else:
        restored_style = cached_style or {'flex': '1 1 25%', 'overflow': 'auto'}
        return restored_style, new_state, html.I(className='fas fa-expand fa-xs'), 'outline-secondary', no_update

# Callback to toggle fullscreen mode for the Prompt Enhancement card
@callback(
    [Output('prompt-enh-card', 'style'),
     Output('prompt-enh-fullscreen', 'data'),
     Output('prompt-enh-expand-btn', 'children'),
     Output('prompt-enh-expand-btn', 'color'),
     Output('prompt-enh-card-style', 'data')],
    Input('prompt-enh-expand-btn', 'n_clicks'),
    State('prompt-enh-fullscreen', 'data'),
    State('prompt-enh-card', 'style'),
    State('prompt-enh-card-style', 'data'),
    prevent_initial_call=True
)
def toggle_prompt_enh_fullscreen(n_clicks, is_fullscreen, current_style, cached_style):
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
            'zIndex': 2000, 'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        return style, new_state, html.I(className='fas fa-compress'), 'secondary', current_style
    else:
        restored_style = cached_style or {'flex': '1 1 25%', 'overflow': 'auto'}
        return restored_style, new_state, html.I(className='fas fa-expand fa-xs'), 'outline-secondary', no_update

# Callback to toggle fullscreen mode for the Rank-Δ card
@callback(
    [Output('rank-delta-card', 'style'),
     Output('rank-delta-fullscreen', 'data'),
     Output('rank-delta-expand-btn', 'children'),
     Output('rank-delta-expand-btn', 'color'),
     Output('rank-delta-card-style', 'data')],
    Input('rank-delta-expand-btn', 'n_clicks'),
    State('rank-delta-fullscreen', 'data'),
    State('rank-delta-card', 'style'),
    State('rank-delta-card-style', 'data'),
    prevent_initial_call=True
)
def toggle_rank_delta_fullscreen(n_clicks, is_fullscreen, current_style, cached_style):
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
            'zIndex': 2000, 'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        return style, new_state, html.I(className='fas fa-compress'), 'secondary', current_style
    else:
        restored_style = cached_style or {'flex': '1 1 25%', 'overflow': 'auto'}
        return restored_style, new_state, html.I(className='fas fa-expand fa-xs'), 'outline-secondary', no_update

# Callback to toggle fullscreen mode for the Saliency card
@callback(
    [Output('saliency-card', 'style'),
     Output('saliency-fullscreen', 'data'),
     Output('saliency-expand-btn', 'children'),
     Output('saliency-expand-btn', 'color'),
     Output('saliency-card-style', 'data')],
    Input('saliency-expand-btn', 'n_clicks'),
    State('saliency-fullscreen', 'data'),
    State('saliency-card', 'style'),
    State('saliency-card-style', 'data'),
    prevent_initial_call=True
)
def toggle_saliency_fullscreen(n_clicks, is_fullscreen, current_style, cached_style):
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
            'zIndex': 2000, 'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        return style, new_state, html.I(className='fas fa-compress'), 'secondary', current_style
    else:
        restored_style = cached_style or {
            'flex': '1 1 25%',
            'display': 'flex',
            'flexDirection': 'column',
            'minHeight': '280px'
        }
        return restored_style, new_state, html.I(className='fas fa-expand fa-xs'), 'outline-secondary', no_update

# Callback to toggle fullscreen mode for the Token Attribution card
@callback(
    [Output('token-attr-card', 'style'),
     Output('token-attr-fullscreen', 'data'),
     Output('token-attr-expand-btn', 'children'),
     Output('token-attr-expand-btn', 'color'),
     Output('token-attr-card-style', 'data')],
    Input('token-attr-expand-btn', 'n_clicks'),
    State('token-attr-fullscreen', 'data'),
    State('token-attr-card', 'style'),
    State('token-attr-card-style', 'data'),
    prevent_initial_call=True
)
def toggle_token_attr_fullscreen(n_clicks, is_fullscreen, current_style, cached_style):
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen
    if new_state:
        style = {
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
            'zIndex': 2000, 'backgroundColor': 'white', 'overflow': 'auto', 'padding': '1rem'
        }
        return style, new_state, html.I(className='fas fa-compress'), 'secondary', current_style
    else:
        restored_style = cached_style or {
            'flex': '1 1 25%',
            'display': 'flex',
            'flexDirection': 'column',
            'minHeight': '280px'
        }
        return restored_style, new_state, html.I(className='fas fa-expand fa-xs'), 'outline-secondary', no_update

# Callback to toggle fullscreen mode for the CIR Controls card
@callback(
    [Output('cir-controls-card', 'style'),
     Output('cir-controls-fullscreen', 'data'),
     Output('cir-controls-expand-btn', 'children'),
     Output('cir-controls-expand-btn', 'color'),
     Output('cir-controls-card-style', 'data')],
    Input('cir-controls-expand-btn', 'n_clicks'),
    State('cir-controls-fullscreen', 'data'),
    State('cir-controls-card', 'style'),
    State('cir-controls-card-style', 'data'),
    prevent_initial_call=True
)
def toggle_cir_controls_fullscreen(n_clicks, is_fullscreen, current_style, cached_style):
    """Toggle fullscreen for the CIR Controls panel."""
    is_fullscreen = bool(is_fullscreen)
    new_state = not is_fullscreen

    if new_state:
        style = {
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'width': '100vw',
            'height': '100vh',
            'zIndex': 2000,
            'backgroundColor': 'white',
            'overflow': 'auto',
            'padding': '1rem'
        }
        icon = html.I(className='fas fa-compress')
        color = 'secondary'
        return style, new_state, icon, color, current_style
    else:
        restored_style = cached_style or {'flex': '1 1 25%', 'overflow': 'auto'}
        icon = html.I(className='fas fa-expand fa-xs')
        color = 'outline-secondary'
        return restored_style, new_state, icon, color, no_update