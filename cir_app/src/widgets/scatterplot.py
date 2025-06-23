from PIL import Image
from dash import dcc
import plotly.express as px
from src.Dataset import Dataset
from src import config
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_marker_colors(trace, colors):
    """Update only the colour of the existing marker dict while preserving
    all other styling attributes such as *size*, *opacity*, *symbol*, etc.

    Args:
        trace (dict): A Plotly trace (typically scatter) in JSON format.
        colors: Either a single colour or an arraylike of colours with the
                 same length as the trace data points.
    """

    # Plotly may return the marker as a *go.scatter.marker* object or as a
    # plain dict.  In both cases ``trace['marker']`` behaves like a mapping.
    marker = dict(trace.get('marker', {}))  # copy to avoid mutating in place
    marker['color'] = colors
    trace['marker'] = marker

def highlight_class_on_scatterplot(scatterplot, class_names):
    """
    Highlight specific classes on the scatterplot.
    
    Args:
        scatterplot: Plotly figure dictionary
        class_names: List of class names to highlight
    """
    if class_names:
        colors = Dataset.get()['class_name'].map(
            lambda x: config.SCATTERPLOT_SELECTED_COLOR if x in class_names else config.SCATTERPLOT_COLOR
        )
    else:
        colors = config.SCATTERPLOT_COLOR
    _set_marker_colors(scatterplot['data'][0], colors)

    # Update legend for selected class based on whether any class is highlighted
    _update_legend_for_selected_class(
        scatterplot,
        class_highlighted=bool(class_names),
        color=config.SCATTERPLOT_SELECTED_COLOR,
    )

def highlight_selected_image_and_class(scatterplot, selected_image_id, class_names):
    """
    Highlight a specific selected image and its class with different colors.
    
    Args:
        scatterplot: Plotly figure dictionary
        selected_image_id: ID of the selected image
        class_names: List of class names to highlight
    """
    df = Dataset.get()
    
    # Check if CIR traces are active
    has_cir_traces = any(trace.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query'] 
                        for trace in scatterplot['data'])
    
    if has_cir_traces:
        # Handle CIR visualization mode: manage traces properly
        _handle_cir_image_selection(scatterplot, selected_image_id, class_names)
    else:
        # Handle regular mode: just update main trace colors
        def get_color(row):
            if row.name == selected_image_id:
                return config.SELECTED_IMAGE_COLOR  # Green for selected image
            elif row['class_name'] in class_names:
                return config.SELECTED_CLASS_COLOR  # Red for same class
            else:
                return config.SCATTERPLOT_COLOR  # Default for others
        
        colors = df.apply(get_color, axis=1)
        _set_marker_colors(scatterplot['data'][0], colors)
    
    # Update legend traces: first selected image, then (optionally) selected class
    _update_legend_for_selected_image(scatterplot)
    _update_legend_for_selected_class(
        scatterplot,
        class_highlighted=bool(class_names),
        color=config.SELECTED_CLASS_COLOR,
    )

def _handle_cir_image_selection(scatterplot, selected_image_id, class_names):
    """
    Handle image selection when CIR traces are active.
    
    Args:
        scatterplot: Plotly figure dictionary
        selected_image_id: ID of the selected image
        class_names: List of class names to highlight
    """
    df = Dataset.get()
    
    # Find main data trace and CIR traces
    main_trace = scatterplot['data'][0]
    xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
    
    # Separate CIR traces and other traces
    cir_traces = []
    other_traces = []
    selected_trace = None
    
    for i, trace in enumerate(scatterplot['data']):
        if i == 0:
            continue  # Skip main trace
        elif trace.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query']:
            cir_traces.append(trace)
        elif trace.get('name') == 'Selected Image':
            selected_trace = trace
        else:
            other_traces.append(trace)
    
    # Find coordinates for the selected image
    selected_x, selected_y = None, None
    for xi, yi, idx in zip(xs, ys, cds):
        if int(idx) == selected_image_id:
            selected_x, selected_y = xi, yi
            break
    
    if selected_x is None:
        return  # Selected image not found in data
    
    # Keep points in their CIR traces; just overlay selected point
    selected_was_top1 = any(
        trace.get('name') == 'Top-1' and len(trace['x']) == 1 and
        abs(trace['x'][0]-selected_x)<1e-10 and abs(trace['y'][0]-selected_y)<1e-10
        for trace in cir_traces
    )
    
    # Rebuild scatterplot data with modified CIR traces
    scatterplot['data'] = [main_trace] + cir_traces + other_traces
    
    # Add new selected image trace
    selected_trace = go.Scatter(
        x=[selected_x], 
        y=[selected_y], 
        mode='markers', 
        marker=dict(color=config.SELECTED_IMAGE_COLOR, size=9), 
        name='Selected Image'
    )
    scatterplot['data'].append(selected_trace.to_plotly_json())
    
    # Update main trace colors for class highlighting
    def get_color(row):
        if row.name == selected_image_id:
            return config.SCATTERPLOT_COLOR  # Selected image handled by separate trace
        elif row['class_name'] in class_names:
            return config.SELECTED_CLASS_COLOR  # Red for same class
        else:
            return config.SCATTERPLOT_COLOR  # Default for others
    
    colors = df.apply(get_color, axis=1)
    _set_marker_colors(scatterplot['data'][0], colors)

def _update_legend_for_selected_image(scatterplot):
    """Update legend to include selected image color when an image is selected."""
    
    # Identify and preserve CIR traces and Selected Image trace
    cir_traces = []
    selected_image_trace = None
    
    for i, trace in enumerate(scatterplot['data']):
        if i == 0:
            # Main data trace - keep it
            continue
        elif trace.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query']:
            # CIR visualization traces - preserve them
            cir_traces.append(trace)
        elif trace.get('name') == 'Selected Image':
            # Selected image trace - preserve it
            selected_image_trace = trace
        else:
            # Legend traces - we'll replace these
            pass
    
    # Remove all traces except the main data trace
    scatterplot['data'] = scatterplot['data'][:1]
    
    # Re-add CIR traces
    scatterplot['data'].extend(cir_traces)
    
    # Re-add selected image trace if it exists
    if selected_image_trace:
        scatterplot['data'].append(selected_image_trace)
    
    # Add updated legend traces only if they don't exist
    # Only count legend traces, not the main data trace (index 0)
    has_embedding_legend = any(i > 0 and trace.get('name') == 'image embedding' 
                              for i, trace in enumerate(scatterplot['data']))
    if not has_embedding_legend:
        scatterplot['data'].append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name='image embedding',
                marker=dict(size=7, color="blue", symbol='circle'),
            ).to_plotly_json()
        )

    # Only add "Selected Image" legend trace if there's no actual selected image trace
    if not selected_image_trace:
        scatterplot['data'].append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name='Selected Image',  # Match the data trace name exactly
                marker=dict(size=7, color=config.SELECTED_IMAGE_COLOR, symbol='circle'),
            ).to_plotly_json()
        )

# ---------------------------------------------------------------------------
# Legend helper for the "selected class" trace
# ---------------------------------------------------------------------------

def _update_legend_for_selected_class(scatterplot, class_highlighted: bool, color: str):
    """
    Add or remove the 'selected class' legend trace.

    Args:
        scatterplot: Plotly figure JSON
        class_highlighted: Whether a class is currently highlighted on the scatterplot
        color: Colour to use for the legend marker when a class is highlighted
    """
    # First, remove any existing 'selected class' legend traces
    scatterplot['data'] = [trace for trace in scatterplot['data'] if trace.get('name') != 'selected class']

    if class_highlighted:
        # Append new legend trace at the end so it shows up in the legend
        scatterplot['data'].append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name='selected class',
                marker=dict(size=7, color=color, symbol='circle'),
            ).to_plotly_json()
        )

# ---------------------------------------------------------------------------
# Zoom-responsive marker sizing functionality
# ---------------------------------------------------------------------------

def calculate_zoom_factor(layout, initial_range=None):
    """
    Calculate zoom factor based on current axis ranges.
    
    Args:
        layout: Plotly figure layout
        initial_range: Dictionary with initial ranges {'x': [min, max], 'y': [min, max]}
                      If None, will calculate from actual data
    
    Returns:
        float: Zoom factor (1.0 = no zoom, >1.0 = zoomed in, <1.0 = zoomed out)
    """
    try:
        # Get current ranges
        current_x_range = layout.get('xaxis', {}).get('range', None)
        current_y_range = layout.get('yaxis', {}).get('range', None)
        
        if not current_x_range or not current_y_range:
            return 1.0
            
        # If no initial range provided, calculate from actual data
        if initial_range is None:
            try:
                from src.Dataset import Dataset
                df = Dataset.get()
                
                # Determine which projection we're using from axis labels
                x_title = layout.get('xaxis', {}).get('title', {})
                if isinstance(x_title, dict):
                    x_title = x_title.get('text', '')
                
                if 'umap' in str(x_title).lower():
                    x_col, y_col = 'umap_x', 'umap_y'
                elif 'tsne' in str(x_title).lower():
                    x_col, y_col = 'tsne_x', 'tsne_y'
                else:
                    # Default fallback ranges
                    initial_range = {'x': [-50, 50], 'y': [-50, 50]}
                    
                if initial_range is None:
                    # Calculate actual data ranges with some padding
                    x_min, x_max = df[x_col].min(), df[x_col].max()
                    y_min, y_max = df[y_col].min(), df[y_col].max()
                    
                    # Add 5% padding
                    x_padding = (x_max - x_min) * 0.05
                    y_padding = (y_max - y_min) * 0.05
                    
                    initial_range = {
                        'x': [x_min - x_padding, x_max + x_padding],
                        'y': [y_min - y_padding, y_max + y_padding]
                    }
                    
            except Exception as e:
                print(f"Error getting data ranges: {e}")
                # Fallback to reasonable defaults
                initial_range = {'x': [-50, 50], 'y': [-50, 50]}
        
        # Calculate zoom factor based on range reduction
        current_x_span = abs(current_x_range[1] - current_x_range[0])
        current_y_span = abs(current_y_range[1] - current_y_range[0])
        
        initial_x_span = abs(initial_range['x'][1] - initial_range['x'][0])
        initial_y_span = abs(initial_range['y'][1] - initial_range['y'][0])
        
        # Use the minimum zoom factor from either axis
        x_zoom = initial_x_span / current_x_span if current_x_span > 0 else 1.0
        y_zoom = initial_y_span / current_y_span if current_y_span > 0 else 1.0
        
        zoom_factor = min(x_zoom, y_zoom)
        
        # Clamp zoom factor to reasonable bounds
        zoom_factor = max(0.1, min(zoom_factor, 20.0))
        
        return zoom_factor
        
    except Exception as e:
        print(f"Error calculating zoom factor: {e}")
        return 1.0

def calculate_marker_size(base_size, zoom_factor, size_type='main'):
    """
    Calculate responsive marker size based on zoom level.
    
    Args:
        base_size: Original marker size
        zoom_factor: Zoom factor from calculate_zoom_factor
        size_type: Type of marker ('main', 'legend', 'cir_trace')
    
    Returns:
        float: Adjusted marker size
    """
    try:
        # Different scaling strategies for different marker types
        if size_type == 'main':
            # Main data points: more aggressive scaling
            # Scale factor: sqrt to make growth less aggressive at high zoom
            scale_factor = max(1.0, min(zoom_factor ** 0.5, 5.0))
            new_size = base_size * scale_factor
        elif size_type == 'cir_trace':
            # CIR traces (Top-K, Top-1, Query, etc.): moderate scaling
            scale_factor = max(1.0, min(zoom_factor ** 0.3, 3.0))
            new_size = base_size * scale_factor
        elif size_type == 'legend':
            # Legend traces: minimal scaling to keep legend readable
            scale_factor = max(1.0, min(zoom_factor ** 0.2, 2.0))
            new_size = base_size * scale_factor
        else:
            # Default: moderate scaling
            scale_factor = max(1.0, min(zoom_factor ** 0.4, 4.0))
            new_size = base_size * scale_factor
        
        # Ensure minimum size
        new_size = max(new_size, 1.0)
        
        return new_size
        
    except Exception as e:
        print(f"Error calculating marker size: {e}")
        return base_size

def apply_zoom_responsive_sizing(scatterplot_fig, zoom_factor):
    """
    Apply zoom-responsive sizing to all traces in the scatterplot.
    
    Args:
        scatterplot_fig: Plotly figure JSON
        zoom_factor: Zoom factor from calculate_zoom_factor
    
    Returns:
        Modified figure with updated marker sizes
    """
    try:
        if not scatterplot_fig or 'data' not in scatterplot_fig:
            return scatterplot_fig
            
        # Base sizes for different trace types
        base_sizes = {
            'main': 2,          # Main data trace
            'legend': 7,        # Legend traces
            'top_k': 7,         # Top-K CIR results
            'top_1': 9,         # Top-1 CIR result
            'query': 12,        # Query marker
            'final_query': 10,  # Final query marker
        }
        
        for i, trace in enumerate(scatterplot_fig['data']):
            trace_name = trace.get('name', '')
            
            # Determine trace type and base size
            if i == 0:
                # Main data trace (always first)
                size_type = 'main'
                base_size = base_sizes['main']
            elif trace_name == 'Top-K':
                size_type = 'cir_trace'
                base_size = base_sizes['top_k']
            elif trace_name == 'Top-1':
                size_type = 'cir_trace'
                base_size = base_sizes['top_1']
            elif trace_name == 'Query':
                size_type = 'cir_trace'
                base_size = base_sizes['query']
            elif trace_name == 'Final Query':
                size_type = 'cir_trace'
                base_size = base_sizes['final_query']
            elif trace_name in ['image embedding', 'selected class', 'Selected Image']:
                size_type = 'legend'
                base_size = base_sizes['legend']
            else:
                # Default handling for other traces
                size_type = 'legend'
                current_size = trace.get('marker', {}).get('size', base_sizes['legend'])
                base_size = current_size
            
            # Calculate new size
            new_size = calculate_marker_size(base_size, zoom_factor, size_type)
            
            # Update marker size
            if 'marker' not in trace:
                trace['marker'] = {}
            trace['marker']['size'] = new_size
            
        return scatterplot_fig
        
    except Exception as e:
        print(f"Error applying zoom responsive sizing: {e}")
        return scatterplot_fig

# ---------------------------------------------------------------------------
# Existing functions continue below
# ---------------------------------------------------------------------------

def add_images_to_scatterplot(scatterplot_fig):
    """Add images to scatterplot when zoomed in"""
    scatterplot_fig['layout']['images'] = []
    scatterplot_data = scatterplot_fig['data'][0]
    scatter_image_ids = scatterplot_data['customdata']
    scatter_x = scatterplot_data['x']
    scatter_y = scatterplot_data['y']

    min_x, max_x = scatterplot_fig['layout']['xaxis']['range']
    min_y, max_y = scatterplot_fig['layout']['yaxis']['range']

    images_in_zoom = []
    for x, y, image_id in zip(scatter_x, scatter_y, scatter_image_ids):
        if min_x <= x <= max_x and min_y <= y <= max_y:
            images_in_zoom.append((x, y, image_id))
        if len(images_in_zoom) > config.MAX_IMAGES_ON_SCATTERPLOT:
            return scatterplot_fig

    if images_in_zoom:
        for x, y, image_id in images_in_zoom:
            image_path = Dataset.get().loc[image_id]['image_path']
            scatterplot_fig['layout']['images'].append(dict(
                x=x,
                y=y,
                source=Image.open(image_path),
                xref="x",
                yref="y",
                sizex=.05,
                sizey=.05,
                xanchor="center",
                yanchor="middle",
            ))
        return scatterplot_fig
    return scatterplot_fig

def create_scatterplot_figure(projection):
    """Create scatterplot figure based on projection type (using go.Scatter)."""
    df = Dataset.get()

    if projection == 't-SNE':
        x_col, y_col = 'tsne_x', 'tsne_y'
    elif projection == 'UMAP':
        x_col, y_col = 'umap_x', 'umap_y'
    else:
        raise Exception('Projection not found')


    # Main scatter trace
    scatter_trace = go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        name='image embedding',
        customdata=df.index,
        showlegend=False,
        marker=dict(
            color=config.SCATTERPLOT_COLOR,
            size=2,
            opacity=0.6,
            symbol='circle'
        ),
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=0.6))
    )

    # Create figure
    fig = go.Figure(data=[scatter_trace])

    fig.update_layout(
        dragmode='select',
        xaxis_title=x_col,
        yaxis_title=y_col,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='image embedding',
            marker=dict(size=7, color="blue", symbol='circle'),
        ))

    return fig

def create_scatterplot(projection):
    """Create scatterplot component"""
    return dcc.Graph(
        figure=create_scatterplot_figure(projection),
        id='scatterplot',
        className='stretchy-widget border-widget',
        responsive=True,
        config={
            'displaylogo': False,
            'modeBarButtonsToRemove': ['autoscale'],
            'displayModeBar': True,
        }
    )

def get_data_selected_on_scatterplot(scatterplot_fig):
    """Get selected data from scatterplot"""
    scatterplot_fig_data = scatterplot_fig['data'][0]

    if 'selectedpoints' in scatterplot_fig_data and scatterplot_fig_data['selectedpoints']:
        selected_image_ids = list(map(scatterplot_fig_data['customdata'].__getitem__, scatterplot_fig_data['selectedpoints']))
        data_selected = Dataset.get().loc[selected_image_ids]
    else:
        # Return empty DataFrame when no selection instead of entire dataset
        data_selected = Dataset.get().iloc[:0]

    # --------------------------------------------------------------
    # Skip adding thumbnail overlays when CIR visualisation markers
    # are present.  Otherwise the images cover the coloured Top-K /
    # Top-1 / Query markers a moment after they are drawn.
    # --------------------------------------------------------------
    if any(tr.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query']
           for tr in scatterplot_fig.get('data', [])):
        return data_selected

    return data_selected 