import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc
import pandas as pd
from src.Dataset import Dataset

def create_histogram(selected_data=None):
    """Create histogram component"""
    histogram = draw_histogram(selected_data)
    return html.Div([
        dcc.Graph(figure=histogram,
                  responsive=True,
                  config={
                      'displaylogo': False,
                      'displayModeBar': False
                  },
                  id='histogram',
                  clear_on_unhover=True),
        dcc.Tooltip(id="histogram-tooltip",
                    loading_text="LOADING"),
    ], className='border-widget stretchy-widget histogram-container')

def draw_histogram(selected_data):
    """Draw histogram showing class distribution"""
    if selected_data is None or len(selected_data) == 0:
        fig = go.Figure()

        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="Select data on the scatterplot",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=28, color="gray")
                )
            ],
            margin=dict(b=0, l=0, r=0, t=100)
        )
        return fig

    df = Dataset.get().loc[selected_data.index]

    # Group by class name and count occurrences
    class_counts = df['class_name'].value_counts().reset_index()
    class_counts.columns = ['class_name', 'count']

    # Create histogram with Plotly Express
    fig = px.histogram(class_counts, x='class_name', y='count')
    fig.update_xaxes(categoryorder='total descending')  # Sort by count

    fig.update_layout(
        xaxis=dict(
            side='top', 
            tickangle=280, 
            automargin=False, 
            fixedrange=True,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            visible=True, 
            automargin=False, 
            fixedrange=True,
            title=dict(text="Frequency", standoff=10),
            tickfont=dict(size=12)
        ),
        margin=dict(l=60, r=60, t=200, b=30)
    )
    
    return fig 