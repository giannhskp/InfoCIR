import dash
from dash import callback, Output, Input, ALL, no_update, State, clientside_callback, ClientsideFunction
from dash.exceptions import PreventUpdate
from src.widgets import scatterplot
import plotly.graph_objects as go

@callback(
    [Output('scatterplot', 'figure', allow_duplicate=True),
     Output('selected-image-data', 'data'),
     Output('selected-gallery-image-ids', 'data')],
    [State('scatterplot', 'figure'),
     State('selected-image-data', 'data'),
     State('selected-gallery-image-ids', 'data')],
    Input({'type': 'gallery-card', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def gallery_image_is_clicked(scatterplot_fig, selected_image_data, current_selected_ids, n_clicks):
    """Handle gallery image clicks to support multi-selection and highlight on scatterplot"""
    if all(e is None for e in n_clicks):
        return no_update, no_update, no_update

    print('Gallery is clicked')
    triggered_id = dash.callback_context.triggered_id['index']
    print(f"Triggered ID: {triggered_id}")
    print(f"Current selected IDs: {current_selected_ids}")
    
    # Initialize current selection if None
    if current_selected_ids is None:
        current_selected_ids = []
    
    # Check if clicking an already selected image (toggle removal)
    if triggered_id in current_selected_ids:
        print("Removing image from selection")
        new_selected_ids = [id for id in current_selected_ids if id != triggered_id]
        
        # If no images left selected, clear everything
        if not new_selected_ids:
            _deselect_all_images(scatterplot_fig, selected_image_data)
            return scatterplot_fig, None, []
        else:
            # Update selection with remaining images
            _update_multi_selection(scatterplot_fig, new_selected_ids, selected_image_data)
            return scatterplot_fig, selected_image_data, new_selected_ids
    
    try:
        # Parse the new string format
        if triggered_id.startswith("image_"):
            # Format: "image_{image_id}_{class_name}"
            parts = triggered_id.split("_", 2)  # Split into at most 3 parts
            if len(parts) >= 3:
                image_id = int(parts[1])
                class_name = parts[2]
                print(f"Adding image ID: {image_id}, class: {class_name} to selection")
                
                # Add to selection
                new_selected_ids = current_selected_ids + [triggered_id]
                
                # Update scatterplot with multi-selection
                _update_multi_selection(scatterplot_fig, new_selected_ids, selected_image_data)
                
                return scatterplot_fig, selected_image_data, new_selected_ids
            else:
                raise ValueError(f"Invalid image identifier format: {triggered_id}")
        elif triggered_id.startswith("class_"):
            # Format: "class_{class_name}" (backwards compatibility)
            class_name = triggered_id[6:]  # Remove "class_" prefix
            print(f"Selected class: {class_name}")
            scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
            return scatterplot_fig, None, current_selected_ids
        else:
            # Legacy format: just the class name directly
            class_name = triggered_id
            print(f"Selected class (legacy): {class_name}")
            scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
            return scatterplot_fig, None, current_selected_ids
            
    except Exception as e:
        print(f"Error in gallery callback: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: try to extract class name and highlight just the class
        if triggered_id.startswith("image_"):
            parts = triggered_id.split("_", 2)
            if len(parts) >= 3:
                class_name = parts[2]
                scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
        elif triggered_id.startswith("class_"):
            class_name = triggered_id[6:]
            scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
        else:
            scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [str(triggered_id)])
    
        return scatterplot_fig, no_update, current_selected_ids


def _update_multi_selection(scatterplot_fig, selected_ids, selected_image_data):
    """Update scatterplot to highlight multiple selected images and their classes"""
    from src.Dataset import Dataset
    
    if not selected_ids:
        _deselect_all_images(scatterplot_fig, selected_image_data)
        return
    
    # Extract image IDs and class names from selected identifiers
    selected_image_ids = []
    selected_class_names = set()
    
    for selected_id in selected_ids:
        if selected_id.startswith("image_"):
            parts = selected_id.split("_", 2)
            if len(parts) >= 3:
                image_id = int(parts[1])
                class_name = parts[2]
                selected_image_ids.append(image_id)
                selected_class_names.add(class_name)
    
    print(f"Multi-selection: {len(selected_image_ids)} images, classes: {selected_class_names}")
    
    # Check if CIR traces are active
    has_cir_traces = any(trace.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query'] 
                        for trace in scatterplot_fig['data'])
    
    if has_cir_traces:
        # Handle CIR visualization mode
        _handle_cir_multi_selection(scatterplot_fig, selected_image_ids, selected_class_names, selected_image_data)
    else:
        # Handle regular mode
        _handle_regular_multi_selection(scatterplot_fig, selected_image_ids, selected_class_names)
    
    # Update legend
    _update_legend_for_multi_selection(scatterplot_fig)


def _handle_regular_multi_selection(scatterplot_fig, selected_image_ids, selected_class_names):
    """Handle multi-selection in regular (non-CIR) mode"""
    from src.Dataset import Dataset
    from src import config
    
    df = Dataset.get()
    
    def get_color(row):
        if row.name in selected_image_ids:
            return config.SELECTED_IMAGE_COLOR  # Green for selected images
        elif row['class_name'] in selected_class_names:
            return config.SELECTED_CLASS_COLOR  # Red for same classes
        else:
            return config.SCATTERPLOT_COLOR  # Default for others
    
    colors = df.apply(get_color, axis=1)
    scatterplot_fig['data'][0]['marker'] = {'color': colors}


def _handle_cir_multi_selection(scatterplot_fig, selected_image_ids, selected_class_names, selected_image_data):
    """Handle multi-selection when CIR traces are active"""
    from src.Dataset import Dataset
    from src import config
    
    df = Dataset.get()
    
    # Find main data trace and CIR traces
    main_trace = scatterplot_fig['data'][0]
    xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
    
    # Separate CIR traces and other traces
    cir_traces = []
    other_traces = []
    
    for i, trace in enumerate(scatterplot_fig['data']):
        if i == 0:
            continue  # Skip main trace
        elif trace.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query']:
            cir_traces.append(trace)
        elif trace.get('name') == 'Selected Images':  # Note: changed to plural
            pass  # Remove existing selected images trace
        else:
            other_traces.append(trace)
    
    # Find coordinates for all selected images
    selected_coordinates = []
    for image_id in selected_image_ids:
        for xi, yi, idx in zip(xs, ys, cds):
            if int(idx) == image_id:
                selected_coordinates.append((xi, yi))
                break
    
    # NOTE: Keep selected images inside Top-K / Top-1 traces so they revert
    # back to the original orange / yellow when deselected.  We only add an
    # extra green overlay; no need to remove them from existing traces.
    
    # Rebuild scatterplot data with modified CIR traces
    scatterplot_fig['data'] = [main_trace] + cir_traces + other_traces
    
    # Add new selected images trace
    if selected_coordinates:
        selected_x = [coord[0] for coord in selected_coordinates]
        selected_y = [coord[1] for coord in selected_coordinates]
        selected_trace = go.Scatter(
            x=selected_x, 
            y=selected_y, 
            mode='markers', 
            marker=dict(color=config.SELECTED_IMAGE_COLOR, size=9), 
            name='Selected Images'
        )
        scatterplot_fig['data'].append(selected_trace.to_plotly_json())
    
    # Update main trace colors for class highlighting
    def get_color(row):
        if row.name in selected_image_ids:
            return config.SCATTERPLOT_COLOR  # Selected images handled by separate trace
        elif row['class_name'] in selected_class_names:
            return config.SELECTED_CLASS_COLOR  # Red for same classes
        else:
            return config.SCATTERPLOT_COLOR  # Default for others
    
    colors = df.apply(get_color, axis=1)
    scatterplot_fig['data'][0]['marker'] = {'color': colors}


def _deselect_all_images(scatterplot_fig, selected_image_data):
    """Deselect all images and restore scatterplot to original state"""
    # Remove any "Selected Images" trace
    _remove_selected_images_trace(scatterplot_fig)
    
    # Check if CIR traces are active
    has_cir_traces = any(trace.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query'] 
                        for trace in scatterplot_fig['data'])
    
    if has_cir_traces and selected_image_data:
        # If we have CIR data, restore any previously selected images to their CIR traces
        _restore_previous_selected_image(scatterplot_fig, selected_image_data)
    
    # Reset the main trace colors to remove all highlighting
    _reset_main_trace_colors(scatterplot_fig)
    
    # Update legend
    _update_legend_for_deselection(scatterplot_fig)


def _remove_selected_images_trace(scatterplot_fig):
    """Remove the 'Selected Images' trace from the scatterplot"""
    traces_to_keep = []
    for trace in scatterplot_fig['data']:
        if trace.get('name') not in ['Selected Image', 'Selected Images']:
            traces_to_keep.append(trace)
    scatterplot_fig['data'] = traces_to_keep


def _update_legend_for_multi_selection(scatterplot_fig):
    """Update legend for multi-selection display"""
    from src import config
    
    # Identify and preserve CIR traces and selected images trace
    cir_traces = []
    selected_images_trace = None
    
    for i, trace in enumerate(scatterplot_fig['data']):
        if i == 0:
            continue  # Main data trace
        elif trace.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query']:
            cir_traces.append(trace)
        elif trace.get('name') == 'Selected Images':
            selected_images_trace = trace
        else:
            pass  # Legend traces - we'll replace these
    
    # Keep only the main data trace and important traces
    scatterplot_fig['data'] = scatterplot_fig['data'][:1] + cir_traces
    if selected_images_trace:
        scatterplot_fig['data'].append(selected_images_trace)
    
    # Add legend traces
    scatterplot_fig['data'].append(
        go.Scatter(
            x=[None], y=[None], mode="markers", name='image embedding',
            marker=dict(size=7, color="blue", symbol='circle'),
        ).to_plotly_json()
    )
    scatterplot_fig['data'].append(
        go.Scatter(
            x=[None], y=[None], mode="markers", name='selected class',
            marker=dict(size=7, color=config.SELECTED_CLASS_COLOR, symbol='circle'),
        ).to_plotly_json()
    )
    
    # Only add "Selected Images" legend if there's no actual trace
    if not selected_images_trace:
        scatterplot_fig['data'].append(
            go.Scatter(
                x=[None], y=[None], mode="markers", name='Selected Images',
                marker=dict(size=7, color=config.SELECTED_IMAGE_COLOR, symbol='circle'),
            ).to_plotly_json()
        )


def _restore_previous_selected_image(scatterplot_fig, selected_image_data):
    """Restore previous selected image to its CIR trace"""
    if not selected_image_data:
        return
    
    prev_x = selected_image_data['x']
    prev_y = selected_image_data['y']
    prev_was_top1 = selected_image_data['was_top1']
    
    # Find CIR traces and restore the previous selected image
    for trace in scatterplot_fig['data']:
        if prev_was_top1 and trace.get('name') == 'Top-1':
            trace['x'] = [prev_x]
            trace['y'] = [prev_y]
            break
        elif not prev_was_top1 and trace.get('name') == 'Top-K':
            if 'x' not in trace or trace['x'] is None:
                trace['x'] = []
                trace['y'] = []
            trace['x'].append(prev_x)
            trace['y'].append(prev_y)
            break


def _reset_main_trace_colors(scatterplot_fig):
    """Reset the main trace colors to default (remove all highlighting)"""
    from src import config
    main_trace = scatterplot_fig['data'][0]  # Main dataset trace
    
    # Reset to default color for all points
    main_trace['marker'] = {'color': config.SCATTERPLOT_COLOR}


def _update_legend_for_deselection(scatterplot_fig):
    """Update legend after deselection to remove selected image references"""
    from src import config
    
    # Identify and preserve CIR traces (Top-K, Top-1, Query, Final Query)
    cir_traces = []
    
    for i, trace in enumerate(scatterplot_fig['data']):
        if i == 0:
            # Main data trace - keep it
            continue
        elif trace.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query']:
            # CIR visualization traces - preserve them
            cir_traces.append(trace)
        else:
            # Legend traces - we'll replace these
            pass
    
    # Keep only the main data trace and CIR traces
    scatterplot_fig['data'] = scatterplot_fig['data'][:1] + cir_traces
    
    # Add basic legend trace for image embeddings
    scatterplot_fig['data'].append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='image embedding',
            marker=dict(size=7, color="blue", symbol='circle'),
        ).to_plotly_json()
    )


# Clientside callback to handle gallery highlighting via direct DOM manipulation (multi-selection)
# Clientside callback to handle gallery highlighting via direct DOM manipulation (multi-selection)
clientside_callback(
    """
    function(selectedImageIdentifiers) {
        console.log('Clientside callback triggered with:', selectedImageIdentifiers);
        console.log('Type of selectedImageIdentifiers:', typeof selectedImageIdentifiers);
        console.log('Is array:', Array.isArray(selectedImageIdentifiers));
        
        // Remove highlighting from all gallery images
        const allImages = document.querySelectorAll('img.gallery-image');
        console.log('Found gallery images:', allImages.length);
        allImages.forEach(img => {
            img.style.border = '';
            img.style.boxShadow = '';
            img.style.transform = '';
            img.style.zIndex = '';
            img.style.borderRadius = '';
        });
        
        // Handle different input formats
        let identifiersToHighlight = [];
        
        if (selectedImageIdentifiers) {
            if (Array.isArray(selectedImageIdentifiers)) {
                identifiersToHighlight = selectedImageIdentifiers;
            } else if (typeof selectedImageIdentifiers === 'string') {
                // Handle case where a single string is passed instead of array
                identifiersToHighlight = [selectedImageIdentifiers];
            } else {
                console.log('Unexpected format for selectedImageIdentifiers:', selectedImageIdentifiers);
                return window.dash_clientside.no_update;
            }
        }
        
        console.log('Identifiers to highlight:', identifiersToHighlight);
        
        // Add highlighting to the selected images
        if (identifiersToHighlight.length > 0) {
            console.log('Highlighting gallery images:', identifiersToHighlight);
            
            const allCards = document.querySelectorAll('a.gallery-card');
            console.log('Found gallery cards:', allCards.length);
            
            identifiersToHighlight.forEach(selectedImageIdentifier => {
                console.log('Processing identifier:', selectedImageIdentifier);
                
                // Find the anchor with matching identifier, then its image child
                let targetCard = null;
                allCards.forEach(card => {
                    const cardId = card.id;
                    console.log('Checking card ID:', cardId, 'against identifier:', selectedImageIdentifier);
                    if (cardId && cardId.includes(selectedImageIdentifier)) {
                        targetCard = card;
                        console.log('Found matching card:', cardId);
                    }
                });
                
                if (targetCard) {
                    const img = targetCard.querySelector('img.gallery-image');
                    if (img) {
                        console.log('Applying highlight to image element:', selectedImageIdentifier);
                        img.style.border = '5px solid #20c997';  // Vibrant green border
                        img.style.boxShadow = '0 0 15px rgba(32, 201, 151, 0.8)'; // Glow effect
                        img.style.transform = 'scale(1.1)'; // Slight zoom
                        img.style.borderRadius = '4px';
                        img.style.zIndex = '10';
                        console.log('Highlight applied successfully');
                    } else {
                        console.log('No img element found in card for:', selectedImageIdentifier);
                    }
                } else {
                    console.log('Card not found for identifier:', selectedImageIdentifier);
                    // Debug: list all available card IDs
                    console.log('Available card IDs:');
                    allCards.forEach(card => {
                        console.log('  -', card.id);
                    });
                }
            });
        } else {
            console.log('No identifiers to highlight');
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output('gallery', 'className'),  # Dummy output to trigger clientside callback
    Input('selected-gallery-image-ids', 'data'),
    prevent_initial_call=True
)