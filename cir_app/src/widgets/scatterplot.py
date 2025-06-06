from PIL import Image
from dash import dcc
import plotly.express as px
from src.Dataset import Dataset
from src import config
import plotly.graph_objects as go

def highlight_class_on_scatterplot(scatterplot, class_names):
    """Highlight specific classes on the scatterplot"""
    if class_names:
        colors = Dataset.get()['class_name'].map(lambda x: config.SCATTERPLOT_SELECTED_COLOR if x in class_names else config.SCATTERPLOT_COLOR)
    else:
        colors = config.SCATTERPLOT_COLOR
    scatterplot['data'][0]['marker'] = {'color': colors}

def highlight_selected_image_and_class(scatterplot, selected_image_id, class_names):
    """Highlight a specific selected image and its class with different colors"""
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
        scatterplot['data'][0]['marker'] = {'color': colors}
    
    # Update legend to include selected image trace (preserve CIR traces)
    _update_legend_for_selected_image(scatterplot)

def _handle_cir_image_selection(scatterplot, selected_image_id, class_names):
    """Handle image selection when CIR traces are active"""
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
    
    # Remove selected image from CIR traces and track which trace it came from
    selected_was_top1 = False
    for trace in cir_traces:
        if trace.get('name') == 'Top-K':
            # Remove selected image from Top-K trace
            new_x, new_y = [], []
            for xi, yi in zip(trace['x'], trace['y']):
                if not (abs(xi - selected_x) < 1e-10 and abs(yi - selected_y) < 1e-10):
                    new_x.append(xi)
                    new_y.append(yi)
            trace['x'] = new_x
            trace['y'] = new_y
            
        elif trace.get('name') == 'Top-1':
            # Check if selected image is the Top-1 and remove it
            if (len(trace['x']) == 1 and 
                abs(trace['x'][0] - selected_x) < 1e-10 and 
                abs(trace['y'][0] - selected_y) < 1e-10):
                trace['x'] = []
                trace['y'] = []
                selected_was_top1 = True
    
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
    scatterplot['data'][0]['marker'] = {'color': colors}

def _update_legend_for_selected_image(scatterplot):
    """Update legend to include selected image color when an image is selected"""
    
    # Identify and preserve CIR traces (Top-K, Top-1, Query, Final Query) and Selected Image trace
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
    
    # Add updated legend traces
    scatterplot['data'].append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='image embedding',
            marker=dict(size=7, color="blue", symbol='circle'),
        ).to_plotly_json()
    )
    scatterplot['data'].append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='selected class',
            marker=dict(size=7, color=config.SELECTED_CLASS_COLOR, symbol='circle'),
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
    """Create scatterplot figure based on projection type"""
    if projection == 't-SNE':
        x_col, y_col = 'tsne_x', 'tsne_y'
    elif projection == 'UMAP':
        x_col, y_col = 'umap_x', 'umap_y'
    else:
        raise Exception('Projection not found')

    fig = px.scatter(data_frame=Dataset.get(), x=x_col, y=y_col)
    fig.update_traces(
        customdata=Dataset.get().index, 
        marker={'color': config.SCATTERPLOT_COLOR},
        unselected_marker_opacity=0.60
    )
    fig.update_layout(dragmode='select')
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    # Add legend traces
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='image embedding',
            marker=dict(size=7, color="blue", symbol='circle'),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='selected class',
            marker=dict(size=7, color="red", symbol='circle'),
        ),
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
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

    if 'selectedpoints' in scatterplot_fig_data:
        selected_image_ids = list(map(scatterplot_fig_data['customdata'].__getitem__, scatterplot_fig_data['selectedpoints']))
        data_selected = Dataset.get().loc[selected_image_ids]
    else:
        data_selected = Dataset.get()

    return data_selected 