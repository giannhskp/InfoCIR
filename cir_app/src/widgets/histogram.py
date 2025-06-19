import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc
import pandas as pd
from src.Dataset import Dataset
from src import config

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

def _truncate_class_name(class_name, max_length=15):
    """Truncate class name if too long"""
    if len(class_name) <= max_length:
        return class_name
    return class_name[:max_length-3] + "..."

def _get_optimal_font_size(num_classes, base_size=10):
    """Calculate optimal font size based on number of classes"""
    if num_classes <= 5:
        return base_size
    elif num_classes <= 10:
        return max(8, base_size - 1)
    elif num_classes <= 15:
        return max(7, base_size - 2)
    else:
        return max(6, base_size - 3)

def draw_histogram(selected_data, highlight_classes=None):
    """Draw histogram showing class distribution.

    Parameters
    ----------
    selected_data : pd.DataFrame | None
        DataFrame containing the subset of data to visualize. When ``None`` an
        informational placeholder figure is returned.
    highlight_classes : list[str] | None
        Optional list of class names that should be highlighted in the bar
        chart. If provided, bars whose underlying ``class_name`` is present in
        the list are coloured with ``config.SELECTED_CLASS_COLOR`` while the
        remaining bars keep the default colour.
    """
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
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="gray")
                )
            ],
            margin=dict(b=20, l=40, r=20, t=40),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

    df = Dataset.get().loc[selected_data.index]

    # Group by class name and count occurrences
    class_counts = df['class_name'].value_counts().reset_index()
    class_counts.columns = ['class_name', 'count']
    
    # Truncate long class names for display
    class_counts['display_name'] = class_counts['class_name'].apply(_truncate_class_name)
    
    # Get optimal font size based on number of classes
    num_classes = len(class_counts)
    font_size = _get_optimal_font_size(num_classes)
    
    # Create histogram with Plotly Express using display names
    fig = px.bar(class_counts, x='display_name', y='count', 
                 hover_data={'class_name': True, 'display_name': False},  # Show full name on hover
                 labels={'display_name': 'Class', 'count': 'Count'})
    
    # Sort by count (descending)
    fig.update_xaxes(categoryorder='total descending')

    # Determine rotation and margins based on number of classes and name lengths
    max_name_length = max(len(name) for name in class_counts['display_name'])
    
    if num_classes <= 3:
        # Few classes: no rotation needed
        tick_angle = 0
        bottom_margin = 60
        top_margin = 40
    elif num_classes <= 6 and max_name_length <= 10:
        # Medium number with short names: slight rotation
        tick_angle = -45
        bottom_margin = 80
        top_margin = 40
    else:
        # Many classes or long names: more rotation
        tick_angle = -60
        bottom_margin = min(120, 60 + max_name_length * 2)
        top_margin = 40

    fig.update_layout(
        xaxis=dict(
            tickangle=tick_angle,
            tickfont=dict(size=font_size),
            title=dict(text="Class", font=dict(size=font_size + 1)),
            fixedrange=True
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(size=font_size + 1)),
            tickfont=dict(size=font_size),
            fixedrange=True
        ),
        margin=dict(l=50, r=20, t=top_margin, b=bottom_margin),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        bargap=0.1,  # Small gap between bars
        # Hover styling
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",  # White background with slight transparency
            bordercolor="rgba(0, 0, 0, 0.2)",     # Light gray border
            font_size=12,                          # Readable font size
            font_family="Arial",                   # Clean font
            font_color="black"                     # Black text for good contrast
        )
    )
    
    # ------------------------------------------------------------------
    # Colour handling – highlight selected classes if requested
    # ------------------------------------------------------------------
    default_bar_colour = 'rgba(31, 119, 180, 0.7)'
    if highlight_classes:
        highlight_set = set(highlight_classes)
        bar_colours = [config.SELECTED_CLASS_COLOR if cn in highlight_set else default_bar_colour
                       for cn in class_counts['class_name']]
    else:
        bar_colours = default_bar_colour  # single colour for all bars

    # Update bar colours and hover template
    fig.update_traces(
        marker_color=bar_colours,
        marker_line_color='rgba(31, 119, 180, 1.0)',
        marker_line_width=1,
        # Custom hover template
        hovertemplate='<b>Class:</b> %{customdata[0]}<br>' +
                      '<b>Count:</b> %{y}<br>' +
                      '<extra></extra>',  # Removes the default trace box
        customdata=class_counts['class_name'].values.reshape(-1, 1)
    )
    
    return fig 