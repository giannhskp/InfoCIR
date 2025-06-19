# src/widgets/tokenbar.py

from dash import html, dcc
import plotly.graph_objects as go

def create_token_bar(token_attr):
    """Create a horizontal bar chart for token attribution"""
    if not token_attr:
        return html.Div("No token attribution available.", className="text-muted")

    tokens, scores = zip(*token_attr)
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=tokens,
        orientation='h',
        marker=dict(color='rgba(58, 71, 80, 0.6)', line=dict(color='rgba(58, 71, 80, 1.0)', width=1))
    ))

    fig.update_layout(
        title='Token Attribution',
        xaxis_title='Importance Score',
        yaxis_title='Token',
        margin=dict(l=60, r=20, t=40, b=40),
        height=400
    )

    return html.Div(
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        className='border-widget stretchy-widget tokenbar-container'
    )
