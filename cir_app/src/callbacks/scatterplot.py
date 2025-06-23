from dash import callback, Output, Input, State, dash, callback_context, ALL
from PIL import Image
from src import config
from src.Dataset import Dataset
from src.widgets import scatterplot
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# UNIFIED SCATTERPLOT CONTROLLER - Single source of truth for figure updates
# ---------------------------------------------------------------------------

@callback(
    Output('scatterplot', 'figure'),
    [
        # Core inputs that affect the figure
        Input('cir-toggle-state', 'data'),
        Input('scatterplot', 'selectedData'),
        Input('scatterplot', 'relayoutData'),
        Input('scatterplot', 'clickData'),
        Input('projection-radio-buttons', 'value'),
        Input('deselect-button', 'n_clicks'),
        Input({'type': 'gallery-card', 'index': ALL}, 'n_clicks'),
        Input('wordcloud', 'click'),
        Input('selected-scatterplot-class', 'data'),
        Input('prompt-selection', 'value'),
        Input('viz-mode', 'data'),
        Input('viz-selected-ids', 'data'),
    ],
    [
        # States needed for decision making
        State('scatterplot', 'figure'),
        State('cir-search-data', 'data'),
        State('selected-gallery-image-ids', 'data'),
        State('selected-image-data', 'data'),
        State('cir-enhanced-prompts-data', 'data'),
        State('selected-histogram-class', 'data'),
    ],
    prevent_initial_call=True,
)
def unified_scatterplot_controller(
    cir_toggle_state, selectedData, relayoutData, clickData, projection_value, deselect_clicks,
    gallery_clicks, wordcloud_click, selected_scatterplot_class, prompt_selection, viz_mode, viz_selected_ids,
    scatterplot_fig, search_data, selected_gallery_image_ids, selected_image_data, 
    enhanced_prompts_data, selected_histogram_class
):
    """
    Unified controller for ALL scatterplot figure updates. This prevents race conditions
    by ensuring only one callback can modify the scatterplot figure.
    """
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    trigger_value = ctx.triggered[0]['value']
    print(f"Scatterplot controller triggered by: {trigger_id} (value: {trigger_value})")
    
    # Debug: Show current figure trace count
    if scatterplot_fig and 'data' in scatterplot_fig:
        current_traces = len(scatterplot_fig['data'])
        current_trace_names = [trace.get('name', 'unnamed') for trace in scatterplot_fig['data']]
        print(f"DEBUG: Input figure has {current_traces} traces: {current_trace_names}")
        
        # Defensive check: If CIR should be active but traces are missing, log warning
        has_cir_traces = any(name in ['Top-K', 'Top-1', 'Query', 'Final Query'] for name in current_trace_names)
        if cir_toggle_state and search_data and not has_cir_traces and current_traces <= 2:
            print(f"WARNING: CIR active but no CIR traces found! Trigger: {trigger_id}")
            print(f"WARNING: This suggests a race condition where CIR traces were lost")
    
    # -------------------------------------------------------------------------
    # 1. PROJECTION CHANGE - Rebuild entire figure from scratch
    # -------------------------------------------------------------------------
    if trigger_id == 'projection-radio-buttons':
        print("Rebuilding scatterplot for projection change")
        new_fig = scatterplot.create_scatterplot_figure(projection_value)
        # Preserve layout if we had one
        if scatterplot_fig and 'layout' in scatterplot_fig:
            new_fig['layout'].update(scatterplot_fig['layout'])
        return new_fig
    
    # -------------------------------------------------------------------------
    # 2. DESELECT BUTTON - Clear selections but preserve CIR traces if active
    # -------------------------------------------------------------------------
    elif trigger_id == 'deselect-button':
        print("Handling deselect button")
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # Clear selections and overlays
        new_fig['layout']['selections'] = None
        new_fig['layout']['images'] = []
        
        # Remove selection traces but keep CIR traces
        new_fig['data'] = [
            trace for trace in new_fig['data']
            if trace.get('name') not in ['Selected Images', 'Selected Image']
        ]
        
        # Clear selectedpoints from main trace
        if new_fig['data'] and 'selectedpoints' in new_fig['data'][0]:
            new_fig['data'][0]['selectedpoints'] = []
            
        # Always reset main trace colors to clear any class highlighting
        from src.widgets.scatterplot import _set_marker_colors
        _set_marker_colors(new_fig['data'][0], config.SCATTERPLOT_COLOR)
        
        # Clear class highlighting legend - remove selected class trace only, don't add legend
        new_fig['data'] = [trace for trace in new_fig['data'] if trace.get('name') != 'selected class']
        
        # Ensure we have exactly one "image embedding" legend trace
        # Only count traces that actually appear in legend (not the main data trace)
        legend_traces = [trace for i, trace in enumerate(new_fig['data']) 
                        if i > 0 and trace.get('name') == 'image embedding']
        has_legend = len(legend_traces) > 0
        if not has_legend:
            # Add legend trace if missing
            legend_trace = go.Scatter(
                x=[None], y=[None], mode="markers", name='image embedding',
                marker=dict(size=7, color="blue", symbol='circle')
            )
            new_fig['data'].append(legend_trace.to_plotly_json())
            
        return new_fig
    
    # -------------------------------------------------------------------------
    # 3. CIR TOGGLE STATE CHANGE - Add/remove result traces
    # -------------------------------------------------------------------------
    elif trigger_id == 'cir-toggle-state':
        print(f"Handling CIR toggle: {cir_toggle_state}")
        print(f"DEBUG: search_data is None: {search_data is None}")
        if search_data:
            print(f"DEBUG: search_data keys: {list(search_data.keys())}")
        
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        if not cir_toggle_state:
            # Hide CIR results - remove all CIR traces but preserve main data and legend traces
            new_fig['layout']['images'] = []
            from src.widgets.scatterplot import _set_marker_colors
            _set_marker_colors(new_fig['data'][0], config.SCATTERPLOT_COLOR)
            
            # Keep main data trace and legend traces, remove only CIR traces
            preserved_traces = []
            has_legend = False
            
            for i, trace in enumerate(new_fig['data']):
                trace_name = trace.get('name', '')
                if trace_name not in ['Top-K', 'Top-1', 'Query', 'Final Query']:
                    preserved_traces.append(trace)
                    # Only count legend traces, not the main data trace (index 0)
                    if trace_name == 'image embedding' and i > 0:
                        has_legend = True
            
            # Ensure we have exactly one legend trace
            if not has_legend:
                legend_trace = go.Scatter(
                    x=[None], y=[None], mode="markers", name='image embedding',
                    marker=dict(size=7, color="blue", symbol='circle')
                )
                preserved_traces.append(legend_trace.to_plotly_json())
            
            new_fig['data'] = preserved_traces
            print(f"DEBUG: After hiding CIR, preserved {len(new_fig['data'])} traces")
            
        elif search_data:
            # Show CIR results - add result traces
            df = Dataset.get()
            topk_ids = search_data.get('topk_ids', [])
            top1_id = search_data.get('top1_id', None)
            
            print(f"DEBUG: Adding CIR traces for {len(topk_ids)} results")
            
            # Clear existing CIR traces first
            new_fig['data'] = [
                trace for trace in new_fig['data']
                if trace.get('name') not in ['Top-K', 'Top-1', 'Query', 'Final Query']
            ]
            new_fig['layout']['images'] = []
            
            # Get coordinates from main trace
            main_trace = new_fig['data'][0]
            xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
            
            # Determine projection type for query positioning
            axis_title = new_fig['layout']['xaxis']['title']['text']
            
            if axis_title == 'umap_x':
                xq, yq = search_data.get('umap_x_query'), search_data.get('umap_y_query')
                xfq, yfq = search_data.get('umap_x_final_query'), search_data.get('umap_y_final_query')
            else:
                xq, yq = search_data.get('tsne_x_query'), search_data.get('tsne_y_query')
                xfq, yfq = None, None  # Final query only for UMAP
            
            # Find coordinates for Top-K and Top-1
            x1, y1, xk, yk = [], [], [], []
            top1_id_cmp = int(top1_id) if top1_id is not None else None
            topk_ids_cmp = [int(x) for x in topk_ids]
            
            for xi, yi, idx in zip(xs, ys, cds):
                idx_cmp = int(idx) if idx is not None else None
                if idx_cmp == top1_id_cmp:
                    x1.append(xi); y1.append(yi)
                elif idx_cmp in topk_ids_cmp:
                    xk.append(xi); yk.append(yi)
            
            # Add traces in order: Top-K, Top-1, Query, Final Query
            if xk:
                trace_k = go.Scatter(
                    x=xk, y=yk, mode='markers', 
                    marker=dict(color=config.TOP_K_COLOR, size=7, opacity=1.0), 
                    name='Top-K',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_k.to_plotly_json())
                print(f"DEBUG: Added Top-K trace with {len(xk)} points, color={config.TOP_K_COLOR}")
                
            if x1:
                trace_1 = go.Scatter(
                    x=x1, y=y1, mode='markers', 
                    marker=dict(color=config.TOP_1_COLOR, size=9, opacity=1.0), 
                    name='Top-1',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_1.to_plotly_json())
                print(f"DEBUG: Added Top-1 trace with {len(x1)} points, color={config.TOP_1_COLOR}")
                
            # Query trace (only for UMAP)
            if xq is not None and axis_title == 'umap_x':
                trace_q = go.Scatter(
                    x=[xq], y=[yq], mode='markers', 
                    marker=dict(color=config.QUERY_COLOR, size=12, symbol='star', opacity=1.0), 
                    name='Query',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_q.to_plotly_json())
                print(f"DEBUG: Added Query trace at ({xq}, {yq}), color={config.QUERY_COLOR}")
                
            # Final Query trace (only for UMAP)
            if xfq is not None and axis_title == 'umap_x':
                trace_fq = go.Scatter(
                    x=[xfq], y=[yfq], mode='markers', 
                    marker=dict(color=config.FINAL_QUERY_COLOR, size=10, symbol='diamond', opacity=1.0), 
                    name='Final Query',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_fq.to_plotly_json())
                print(f"DEBUG: Added Final Query trace at ({xfq}, {yfq}), color={config.FINAL_QUERY_COLOR}")
            
            print(f"DEBUG: Final figure has {len(new_fig['data'])} traces total")
            
            # Debug: Print trace names to verify they're there
            trace_names = [trace.get('name', 'unnamed') for trace in new_fig['data']]
            print(f"DEBUG: Trace names: {trace_names}")
        
        # Ensure we have exactly one "image embedding" legend trace when CIR is active
        # Only count legend traces, not the main data trace (index 0)
        legend_traces = [trace for i, trace in enumerate(new_fig['data']) 
                        if i > 0 and trace.get('name') == 'image embedding']
        has_legend = len(legend_traces) > 0
        if not has_legend:
            # Add legend trace if missing
            legend_trace = go.Scatter(
                x=[None], y=[None], mode="markers", name='image embedding',
                marker=dict(size=7, color="blue", symbol='circle')
            )
            new_fig['data'].append(legend_trace.to_plotly_json())
            print(f"DEBUG: Added missing 'image embedding' legend trace")
        
        # Apply zoom-responsive sizing to all traces including newly added CIR traces
        if 'layout' in new_fig and new_fig['layout'].get('xaxis', {}).get('range'):
            zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
            print(f"Applying zoom factor {zoom_factor} to CIR traces")
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
        
        # Final debug check to ensure CIR traces are properly added
        final_trace_names = [trace.get('name', 'unnamed') for trace in new_fig['data']]
        print(f"DEBUG: cir-toggle-state final trace count: {len(new_fig['data'])}, names: {final_trace_names}")
        
        return new_fig
    
    # -------------------------------------------------------------------------
    # 4. ZOOM/RELAYOUT - Add thumbnail images when zoomed in and apply zoom-responsive sizing
    # -------------------------------------------------------------------------
    elif trigger_id == 'scatterplot' and relayoutData:
        print("Handling scatterplot zoom/relayout")
        print(f"DEBUG: relayoutData = {relayoutData}")
        
        # Skip dragmode changes
        if len(relayoutData) == 1 and 'dragmode' in relayoutData:
            return dash.no_update
        
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # Check if this is an axis reset (autorange) or explicit zoom
        is_axis_reset = ('xaxis.autorange' in relayoutData or 'yaxis.autorange' in relayoutData)
        has_explicit_range = ('xaxis.range[0]' in relayoutData or 'yaxis.range[0]' in relayoutData)
        
        if is_axis_reset:
            print('Handling axis reset - restoring original marker sizes')
            # Axis reset: apply zoom factor of 1.0 (original sizes)
            zoom_factor = 1.0
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
            # Clear thumbnail images when resetting
            new_fig['layout']['images'] = []
            
        elif has_explicit_range:
            print('Adding thumbnail overlays and applying zoom-responsive sizing')
            # Apply zoom-responsive marker sizing
            zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
            print(f"Calculated zoom factor: {zoom_factor}")
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
            
            # Add thumbnail images
            new_fig = scatterplot.add_images_to_scatterplot(new_fig)
        else:
            # Other relayout changes - apply zoom-responsive sizing if we have ranges
            print(f"Other relayout change: {relayoutData}")
            # Don't block the update, but apply zoom-responsive sizing if possible
            if 'layout' in new_fig and new_fig['layout'].get('xaxis', {}).get('range'):
                zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
                print(f"Applying zoom factor {zoom_factor} to other relayout change")
                new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
        
        return new_fig
    
    # -------------------------------------------------------------------------
    # 5. SCATTERPLOT SELECTION - Handle drag selection (only when CIR off)
    # -------------------------------------------------------------------------
    elif trigger_id == 'scatterplot' and selectedData is not None:
        print("Handling scatterplot selection")
        
        # Skip when CIR overlay is active
        if cir_toggle_state:
            return dash.no_update
            
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # Clear thumbnail overlays
        new_fig['layout']['images'] = []
        
        # Reset colors to default (no class highlight)
        scatterplot.highlight_class_on_scatterplot(new_fig, None)
        
        return new_fig
    
    # -------------------------------------------------------------------------
    # 6. GALLERY CLICKS - Handle image selection from gallery
    # -------------------------------------------------------------------------
    elif trigger_id.startswith('{"index"') and trigger_id.endswith('gallery-card"}'):
        print("Handling gallery image selection - returning no_update for now")
        
        # Defensive check: If CIR should be active but traces are missing, restore them
        if scatterplot_fig and 'data' in scatterplot_fig:
            current_trace_names = [trace.get('name', 'unnamed') for trace in scatterplot_fig['data']]
            has_cir_traces = any(name in ['Top-K', 'Top-1', 'Query', 'Final Query'] for name in current_trace_names)
            
            if cir_toggle_state and search_data and not has_cir_traces:
                print("WARNING: Gallery click detected missing CIR traces, restoring them")
                # Restore CIR traces using the same logic as cir-toggle-state handler
                import copy
                new_fig = copy.deepcopy(scatterplot_fig)
                
                # Re-add CIR traces (simplified version)
                df = Dataset.get()
                topk_ids = search_data.get('topk_ids', [])
                top1_id = search_data.get('top1_id', None)
                
                # Get coordinates from main trace
                main_trace = new_fig['data'][0]
                xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
                
                # Find coordinates for Top-K and Top-1
                x1, y1, xk, yk = [], [], [], []
                top1_id_cmp = int(top1_id) if top1_id is not None else None
                topk_ids_cmp = [int(x) for x in topk_ids]
                
                for xi, yi, idx in zip(xs, ys, cds):
                    idx_cmp = int(idx) if idx is not None else None
                    if idx_cmp == top1_id_cmp:
                        x1.append(xi); y1.append(yi)
                    elif idx_cmp in topk_ids_cmp:
                        xk.append(xi); yk.append(yi)
                
                # Add traces in order: Top-K, Top-1, Query, Final Query
                if xk:
                    trace_k = go.Scatter(
                        x=xk, y=yk, mode='markers', 
                        marker=dict(color=config.TOP_K_COLOR, size=7, opacity=1.0), 
                        name='Top-K', showlegend=True, visible=True
                    )
                    new_fig['data'].append(trace_k.to_plotly_json())
                    
                if x1:
                    trace_1 = go.Scatter(
                        x=x1, y=y1, mode='markers', 
                        marker=dict(color=config.TOP_1_COLOR, size=9, opacity=1.0), 
                        name='Top-1', showlegend=True, visible=True
                    )
                    new_fig['data'].append(trace_1.to_plotly_json())
                
                # Add Query and Final Query traces for UMAP
                axis_title = new_fig['layout']['xaxis']['title']['text']
                if axis_title == 'umap_x':
                    xq, yq = search_data.get('umap_x_query'), search_data.get('umap_y_query')
                    xfq, yfq = search_data.get('umap_x_final_query'), search_data.get('umap_y_final_query')
                    
                    if xq is not None:
                        trace_q = go.Scatter(
                            x=[xq], y=[yq], mode='markers', 
                            marker=dict(color=config.QUERY_COLOR, size=12, symbol='star', opacity=1.0), 
                            name='Query', showlegend=True, visible=True
                        )
                        new_fig['data'].append(trace_q.to_plotly_json())
                        
                    if xfq is not None:
                        trace_fq = go.Scatter(
                            x=[xfq], y=[yfq], mode='markers', 
                            marker=dict(color=config.FINAL_QUERY_COLOR, size=10, symbol='diamond', opacity=1.0), 
                            name='Final Query', showlegend=True, visible=True
                        )
                        new_fig['data'].append(trace_fq.to_plotly_json())
                
                # Apply zoom-responsive sizing
                if 'layout' in new_fig and new_fig['layout'].get('xaxis', {}).get('range'):
                    zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
                    new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
                
                print(f"WARNING: Restored CIR traces, now have {len(new_fig['data'])} traces")
                return new_fig
        
        # TODO: Implement proper gallery selection logic
        # For now, don't interfere with scatterplot figure
        return dash.no_update
    
    # -------------------------------------------------------------------------
    # 7. WORDCLOUD CLICKS - Handle class highlighting from wordcloud
    # -------------------------------------------------------------------------
    elif trigger_id == 'wordcloud' and wordcloud_click:
        print("Handling wordcloud click")
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # Extract class name from wordcloud click and highlight
        if 'points' in wordcloud_click and wordcloud_click['points']:
            class_name = wordcloud_click['points'][0].get('text')
            if class_name:
                scatterplot.highlight_class_on_scatterplot(new_fig, [class_name])
        
        return new_fig
    
    # -------------------------------------------------------------------------
    # 8. CLASS SELECTION CHANGES - Handle scatterplot highlighting when class selection changes
    # -------------------------------------------------------------------------
    elif trigger_id == 'selected-scatterplot-class':
        print(f"Handling class selection change: {selected_scatterplot_class}")
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # When CIR is active, be careful not to interfere with CIR traces
        if cir_toggle_state:
            print("DEBUG: Class selection while CIR is active - preserving CIR traces")
            # Only update main trace colors, preserve all CIR traces
            if selected_scatterplot_class is None:
                # Deselect - clear class highlighting but preserve CIR traces
                from src.widgets.scatterplot import _set_marker_colors
                _set_marker_colors(new_fig['data'][0], config.SCATTERPLOT_COLOR)
            else:
                # Select - highlight this class but preserve CIR traces
                df = Dataset.get()
                colors = df['class_name'].map(
                    lambda x: config.SCATTERPLOT_SELECTED_COLOR if x == selected_scatterplot_class else config.SCATTERPLOT_COLOR
                )
                from src.widgets.scatterplot import _set_marker_colors
                _set_marker_colors(new_fig['data'][0], colors)
            
            # Update legend for selected class (preserve CIR traces in legend)
            from src.widgets.scatterplot import _update_legend_for_selected_class
            _update_legend_for_selected_class(
                new_fig,
                class_highlighted=bool(selected_scatterplot_class),
                color=config.SCATTERPLOT_SELECTED_COLOR,
            )
            
            # Ensure we have exactly one "image embedding" legend trace when CIR is active during class selection
            # Only count legend traces, not the main data trace (index 0)
            legend_traces = [trace for i, trace in enumerate(new_fig['data']) 
                            if i > 0 and trace.get('name') == 'image embedding']
            has_legend = len(legend_traces) > 0
            if not has_legend:
                # Add legend trace if missing
                legend_trace = go.Scatter(
                    x=[None], y=[None], mode="markers", name='image embedding',
                    marker=dict(size=7, color="blue", symbol='circle')
                )
                new_fig['data'].append(legend_trace.to_plotly_json())
        else:
            # Normal mode - no CIR traces to preserve
            if selected_scatterplot_class is None:
                # Deselect - clear all class highlighting
                from src.widgets.scatterplot import _set_marker_colors
                _set_marker_colors(new_fig['data'][0], config.SCATTERPLOT_COLOR)
                # Also clear class highlighting legend
                scatterplot.highlight_class_on_scatterplot(new_fig, None)
            else:
                # Select - highlight this class
                scatterplot.highlight_class_on_scatterplot(new_fig, [selected_scatterplot_class])
        
        # Apply zoom-responsive sizing to all traces including class highlighting
        if 'layout' in new_fig and new_fig['layout'].get('xaxis', {}).get('range'):
            zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
            print(f"Applying zoom factor {zoom_factor} to class selection traces")
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
        
        return new_fig
    
    # -------------------------------------------------------------------------
    # 9. ENHANCED PROMPT SELECTION - Handle prompt selection changes
    # -------------------------------------------------------------------------
    elif trigger_id == 'prompt-selection':
        print(f"Handling prompt selection: {prompt_selection}")
        
        if not cir_toggle_state or not enhanced_prompts_data:
            return dash.no_update
            
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # Clear existing CIR traces
        new_fig['data'] = [
            trace for trace in new_fig['data']
            if trace.get('name') not in ['Top-K', 'Top-1', 'Query', 'Final Query']
        ]
        
        # Determine which results to show
        if prompt_selection == -1:
            # Revert to original results
            if search_data:
                topk_ids = search_data.get('topk_ids', [])
                top1_id = search_data.get('top1_id', None)
        elif prompt_selection is not None and prompt_selection >= 0:
            # Show enhanced prompt results
            results_lists = enhanced_prompts_data.get('all_results', [])
            if prompt_selection < len(results_lists):
                results = results_lists[prompt_selection]
                topk_ids = []
                df = Dataset.get()
                for img_name, _ in results:
                    try:
                        idx = int(img_name)
                        if idx in df.index:
                            topk_ids.append(idx)
                    except:
                        if img_name in df.index:
                            topk_ids.append(img_name)
                top1_id = topk_ids[0] if topk_ids else None
            else:
                return dash.no_update
        else:
            return dash.no_update
        
        # Add the appropriate result traces (similar to CIR toggle logic)
        main_trace = new_fig['data'][0]
        xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
        
        x1, y1, xk, yk = [], [], [], []
        top1_id_cmp = int(top1_id) if top1_id is not None else None
        topk_ids_cmp = [int(x) for x in topk_ids]
        
        for xi, yi, idx in zip(xs, ys, cds):
            idx_cmp = int(idx) if idx is not None else None
            if idx_cmp == top1_id_cmp:
                x1.append(xi); y1.append(yi)
            elif idx_cmp in topk_ids_cmp:
                xk.append(xi); yk.append(yi)
        
        # Add traces in order: Top-K, Top-1, Query, Final Query
        if xk:
            trace_k = go.Scatter(
                x=xk, y=yk, mode='markers', 
                marker=dict(color=config.TOP_K_COLOR, size=7, opacity=1.0), 
                name='Top-K',
                showlegend=True,
                visible=True
            )
            new_fig['data'].append(trace_k.to_plotly_json())
            
        if x1:
            trace_1 = go.Scatter(
                x=x1, y=y1, mode='markers', 
                marker=dict(color=config.TOP_1_COLOR, size=9, opacity=1.0), 
                name='Top-1',
                showlegend=True,
                visible=True
            )
            new_fig['data'].append(trace_1.to_plotly_json())
        
        # Add Query and Final Query traces if they exist in search_data
        # Determine projection type for query positioning
        axis_title = new_fig['layout']['xaxis']['title']['text']
        if axis_title == 'umap_x' and search_data:
            xq, yq = search_data.get('umap_x_query'), search_data.get('umap_y_query')
            
            # Query trace (only for UMAP) - always use original query coordinates
            if xq is not None:
                trace_q = go.Scatter(
                    x=[xq], y=[yq], mode='markers', 
                    marker=dict(color=config.QUERY_COLOR, size=12, symbol='star', opacity=1.0), 
                    name='Query',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_q.to_plotly_json())
                
            # Final Query trace (only for UMAP)
            xfq = yfq = None
            if prompt_selection is not None and prompt_selection >= 0:
                # Use enhanced prompt's Final Query coordinates
                enhanced_coords = enhanced_prompts_data.get('enhanced_final_query_coords', [])
                if prompt_selection < len(enhanced_coords):
                    coord_data = enhanced_coords[prompt_selection]
                    xfq, yfq = coord_data.get('x'), coord_data.get('y')
                else:
                    xfq, yfq = search_data.get('umap_x_final_query'), search_data.get('umap_y_final_query')
                    print(f"DEBUG: prompt-selection falling back to original coords: x={xfq}, y={yfq}")
            else:
                # Use original Final Query coordinates
                xfq, yfq = search_data.get('umap_x_final_query'), search_data.get('umap_y_final_query')
                print(f"DEBUG: prompt-selection using original coords: x={xfq}, y={yfq}")
                
            if xfq is not None and yfq is not None:
                trace_fq = go.Scatter(
                    x=[xfq], y=[yfq], mode='markers', 
                    marker=dict(color=config.FINAL_QUERY_COLOR, size=10, symbol='diamond', opacity=1.0), 
                    name='Final Query',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_fq.to_plotly_json())
        
        # Ensure we have exactly one "image embedding" legend trace when enhanced prompt is selected
        # Only count legend traces, not the main data trace (index 0)
        legend_traces = [trace for i, trace in enumerate(new_fig['data']) 
                        if i > 0 and trace.get('name') == 'image embedding']
        has_legend = len(legend_traces) > 0
        if not has_legend:
            # Add legend trace if missing
            legend_trace = go.Scatter(
                x=[None], y=[None], mode="markers", name='image embedding',
                marker=dict(size=7, color="blue", symbol='circle')
            )
            new_fig['data'].append(legend_trace.to_plotly_json())
        
        # Apply zoom-responsive sizing to all traces including enhanced prompt traces
        if 'layout' in new_fig and new_fig['layout'].get('xaxis', {}).get('range'):
            zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
            print(f"Applying zoom factor {zoom_factor} to enhanced prompt traces")
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
        
        return new_fig
    
    # -------------------------------------------------------------------------
    # 10. VISUALIZATION MODE - Handle viz mode selection changes
    # -------------------------------------------------------------------------
    elif trigger_id == 'viz-selected-ids':
        print(f"Handling viz selection: {viz_selected_ids}")
        print(f"DEBUG: viz_mode = {viz_mode}")
        print(f"DEBUG: cir_toggle_state = {cir_toggle_state}")
        print(f"DEBUG: search_data is None = {search_data is None}")
        print(f"DEBUG: prompt_selection = {prompt_selection}")
        
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # CRITICAL: Check if CIR is active and we need to preserve/add CIR traces
        # This handles the race condition where viz-selected-ids and cir-toggle-state 
        # are both triggered simultaneously by the CIR search callback
        # This must happen BEFORE the viz_mode check to handle the race condition
        if cir_toggle_state and search_data:
            print("DEBUG: viz-selected-ids detected CIR is active, preserving/adding CIR traces")
            
            # Remove all non-main traces first
            new_fig['data'] = [new_fig['data'][0]]  # Keep only main data trace
            
            # Re-add CIR traces (same logic as CIR toggle handler)
            df = Dataset.get()
            
            # FIXED: Determine which data to use based on current prompt selection
            if prompt_selection is not None and prompt_selection >= 0 and enhanced_prompts_data:
                # Use enhanced prompt results
                results_lists = enhanced_prompts_data.get('all_results', [])
                if prompt_selection < len(results_lists):
                    results = results_lists[prompt_selection]
                    topk_ids = []
                    for img_name, _ in results:
                        try:
                            idx = int(img_name)
                            if idx in df.index:
                                topk_ids.append(idx)
                        except:
                            if img_name in df.index:
                                topk_ids.append(img_name)
                    top1_id = topk_ids[0] if topk_ids else None
                    print(f"DEBUG: viz-selected-ids using enhanced prompt {prompt_selection} with {len(topk_ids)} results")
                else:
                    # Fallback to original search data
                    topk_ids = search_data.get('topk_ids', [])
                    top1_id = search_data.get('top1_id', None)
                    print(f"DEBUG: viz-selected-ids falling back to original search data")
            else:
                # Use original search data
                topk_ids = search_data.get('topk_ids', [])
                top1_id = search_data.get('top1_id', None)
                print(f"DEBUG: viz-selected-ids using original search data")
            
            # Get coordinates from main trace
            main_trace = new_fig['data'][0]
            xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
            
            # Determine projection type for query positioning
            axis_title = new_fig['layout']['xaxis']['title']['text']
            if axis_title == 'umap_x':
                xq, yq = search_data.get('umap_x_query'), search_data.get('umap_y_query')
                
                # Determine Final Query coordinates based on prompt selection
                xfq = yfq = None
                if prompt_selection is not None and prompt_selection >= 0 and enhanced_prompts_data:
                    # Use enhanced prompt's Final Query coordinates
                    enhanced_coords = enhanced_prompts_data.get('enhanced_final_query_coords', [])
                    if prompt_selection < len(enhanced_coords):
                        coord_data = enhanced_coords[prompt_selection]
                        xfq, yfq = coord_data.get('x'), coord_data.get('y')
                    else:
                        xfq, yfq = search_data.get('umap_x_final_query'), search_data.get('umap_y_final_query')
                else:
                    # Use original Final Query coordinates
                    xfq, yfq = search_data.get('umap_x_final_query'), search_data.get('umap_y_final_query')
            else:
                xq, yq = search_data.get('tsne_x_query'), search_data.get('tsne_y_query')
                xfq, yfq = None, None  # Final query only for UMAP
            
            # Find coordinates for Top-K and Top-1
            x1, y1, xk, yk = [], [], [], []
            top1_id_cmp = int(top1_id) if top1_id is not None else None
            topk_ids_cmp = [int(x) for x in topk_ids]
            
            for xi, yi, idx in zip(xs, ys, cds):
                idx_cmp = int(idx) if idx is not None else None
                if idx_cmp == top1_id_cmp:
                    x1.append(xi); y1.append(yi)
                elif idx_cmp in topk_ids_cmp:
                    xk.append(xi); yk.append(yi)
            
            # Add CIR traces
            if xk:
                trace_k = go.Scatter(
                    x=xk, y=yk, mode='markers', 
                    marker=dict(color=config.TOP_K_COLOR, size=7, opacity=1.0), 
                    name='Top-K',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_k.to_plotly_json())
                
            if x1:
                trace_1 = go.Scatter(
                    x=x1, y=y1, mode='markers', 
                    marker=dict(color=config.TOP_1_COLOR, size=9, opacity=1.0), 
                    name='Top-1',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_1.to_plotly_json())
                
            # Query trace (only for UMAP)
            if xq is not None and axis_title == 'umap_x':
                trace_q = go.Scatter(
                    x=[xq], y=[yq], mode='markers', 
                    marker=dict(color=config.QUERY_COLOR, size=12, symbol='star', opacity=1.0), 
                    name='Query',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_q.to_plotly_json())
                
            # Final Query trace (only for UMAP)
            if xfq is not None and yfq is not None and axis_title == 'umap_x':
                trace_fq = go.Scatter(
                    x=[xfq], y=[yfq], mode='markers', 
                    marker=dict(color=config.FINAL_QUERY_COLOR, size=10, symbol='diamond', opacity=1.0), 
                    name='Final Query',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_fq.to_plotly_json())
            
            print(f"DEBUG: viz-selected-ids re-added CIR traces, now have {len(new_fig['data'])} traces")
            
            # Ensure we have exactly one "image embedding" legend trace when CIR is active
            # Only count legend traces, not the main data trace (index 0)
            legend_traces = [trace for i, trace in enumerate(new_fig['data']) 
                            if i > 0 and trace.get('name') == 'image embedding']
            has_legend = len(legend_traces) > 0
            if not has_legend:
                # Add legend trace if missing
                legend_trace = go.Scatter(
                    x=[None], y=[None], mode="markers", name='image embedding',
                    marker=dict(size=7, color="blue", symbol='circle')
                )
                new_fig['data'].append(legend_trace.to_plotly_json())
                print(f"DEBUG: viz-selected-ids added missing 'image embedding' legend trace")
        else:
            # Remove existing Selected Images trace but preserve other traces
            new_fig['data'] = [tr for tr in new_fig['data'] if tr.get('name') != 'Selected Images']
        
        print(f"DEBUG: viz-selected-ids preserving {len(new_fig['data'])} traces")
        
        # If visualization mode is OFF and we don't have CIR traces to preserve, return no update
        if not viz_mode and not (cir_toggle_state and search_data):
            return dash.no_update
        
        if viz_selected_ids and viz_mode:
            main_trace = new_fig['data'][0]
            xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
            sel_x, sel_y = [], []
            sel_set = set(str(vid) for vid in viz_selected_ids)
            
            for xi, yi, cid in zip(xs, ys, cds):
                if str(cid) in sel_set:
                    sel_x.append(xi)
                    sel_y.append(yi)
            
            if sel_x:
                sel_trace = go.Scatter(
                    x=sel_x, y=sel_y, mode='markers',
                    marker=dict(color=config.SELECTED_IMAGE_COLOR, size=9),
                    name='Selected Images'
                )
                new_fig['data'].append(sel_trace.to_plotly_json())
                print(f"DEBUG: Added Selected Images trace with {len(sel_x)} points")
        
        # Apply zoom-responsive sizing to all traces including visualization mode traces
        if 'layout' in new_fig and new_fig['layout'].get('xaxis', {}).get('range'):
            zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
            print(f"Applying zoom factor {zoom_factor} to visualization mode traces")
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
        
        print(f"DEBUG: viz-selected-ids returning figure with {len(new_fig['data'])} traces")
        return new_fig
    
    # Default: no update
    print(f"No handler for trigger: {trigger_id}")
    return dash.no_update


# ---------------------------------------------------------------------------
# SEPARATE CALLBACKS FOR OTHER WIDGETS (no scatterplot figure output)
# ---------------------------------------------------------------------------

@callback(
    [Output("gallery", "children"),
     Output("wordcloud", "list"),
     Output('histogram', 'figure'),
     Output('selected-image-data', 'data', allow_duplicate=True),
     Output('selected-gallery-image-ids', 'data', allow_duplicate=True)],
    Input('cir-toggle-state', 'data'),
    [State('cir-search-data', 'data'),
     State('selected-gallery-image-ids', 'data')],
    prevent_initial_call=True,
)
def update_widgets_for_cir_toggle(cir_toggle_state, search_data, selected_gallery_image_ids):
    """Update other widgets when CIR toggle changes (no scatterplot figure output)"""
    
    if not cir_toggle_state:
        # Hide action: reset widgets
        gallery_children = []
        wordcloud_data = []
        from src.widgets import histogram
        histogram_fig = histogram.draw_histogram(None)
        return gallery_children, wordcloud_data, histogram_fig, None, []
        
    elif search_data:
        # Visualize action: populate widgets
        from src.widgets import gallery, wordcloud, histogram
        df = Dataset.get()
        topk_ids = search_data.get('topk_ids', [])
        
        # Build wordcloud
        class_counts = df.loc[topk_ids]['class_name'].value_counts()
        if len(class_counts):
            weights = wordcloud.wordcloud_weight_rescale(class_counts.values, 1, class_counts.max())
            wordcloud_data = sorted([[cn, w] for cn, w in zip(class_counts.index, weights)], key=lambda x: x[1], reverse=True)
        else:
            wordcloud_data = []
        
        # Build gallery
        cir_data = df.loc[topk_ids]
        gallery_children = gallery.create_gallery_children(
            cir_data['image_path'].values, 
            cir_data['class_name'].values,
            cir_data.index.values,
            selected_gallery_image_ids
        )
        
        # Build histogram
        histogram_fig = histogram.draw_histogram(cir_data)
        
        return gallery_children, wordcloud_data, histogram_fig, None, selected_gallery_image_ids
    
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    [Output("gallery", "children", allow_duplicate=True),
     Output("wordcloud", "list", allow_duplicate=True),
     Output('histogram', 'figure', allow_duplicate=True),
     Output('selected-image-data', 'data', allow_duplicate=True),
     Output('selected-gallery-image-ids', 'data', allow_duplicate=True)],
    Input('scatterplot', 'selectedData'),
    [State('cir-toggle-state', 'data'),
     State('scatterplot', 'figure')],
    prevent_initial_call=True,
)
def update_widgets_from_selection(selectedData, cir_toggle_state, scatterplot_fig):
    """Update widgets from scatterplot selection (no scatterplot figure output)"""
    
    # When CIR is active, only update if there's an actual user selection
    # This preserves CIR visualization when there's no selection
    if cir_toggle_state:
        # Check if there's an actual selection by looking at the scatterplot figure
        main_trace = scatterplot_fig['data'][0] if scatterplot_fig and scatterplot_fig['data'] else None
        has_selection = (main_trace and 
                        'selectedpoints' in main_trace and 
                        main_trace['selectedpoints'] and 
                        len(main_trace['selectedpoints']) > 0)
        
        if not has_selection:
            # No actual selection when CIR is active - preserve CIR visualization
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    from src.widgets import gallery, wordcloud, histogram
    
    # Get selected data
    data_sel = scatterplot.get_data_selected_on_scatterplot(scatterplot_fig)
    
    # Build wordcloud
    class_counts = data_sel['class_name'].value_counts()
    if len(class_counts):
        weights = wordcloud.wordcloud_weight_rescale(class_counts.values, 1, class_counts.max())
        wordcloud_data = sorted([[cn, w] for cn, w in zip(class_counts.index, weights)], key=lambda x: x[1], reverse=True)
    else:
        wordcloud_data = []
    
    # Sample for gallery
    sample = data_sel.sample(min(len(data_sel), config.IMAGE_GALLERY_SIZE), random_state=1) if len(data_sel) else data_sel
    gallery_children = gallery.create_gallery_children(
        sample['image_path'].values,
        sample['class_name'].values,
        sample.index.values,
        []
    )
    
    histogram_fig = histogram.draw_histogram(data_sel)
    
    return gallery_children, wordcloud_data, histogram_fig, None, []


# ---------------------------------------------------------------------------
# CLASS SELECTION MANAGEMENT - Handle selected class state and histogram highlighting
# ---------------------------------------------------------------------------

@callback(
    [Output('selected-scatterplot-class', 'data'),
     Output('histogram', 'figure', allow_duplicate=True)],
    [Input('scatterplot', 'clickData'),
     Input('histogram', 'clickData')],
    [State('selected-scatterplot-class', 'data'),
     State('scatterplot', 'figure'),
     State('cir-toggle-state', 'data'),
     State('cir-search-data', 'data')],
    prevent_initial_call=True,
)
def manage_class_selection_and_histogram_highlighting(scatterplot_click, histogram_click, 
                                                      current_selected_class, scatterplot_fig, 
                                                      cir_toggle_state, search_data):
    """Manage class selection state and update histogram highlighting accordingly"""
    
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    new_selected_class = current_selected_class
    
    # Handle different triggers (excluding deselect-button which is handled by deselect_button_is_pressed)
    if trigger_id == 'scatterplot' and scatterplot_click:
        # Handle scatterplot click
        if 'points' in scatterplot_click and scatterplot_click['points']:
            clicked_point = scatterplot_click['points'][0]
            if 'customdata' in clicked_point:
                clicked_image_id = clicked_point['customdata']
                df = Dataset.get()
                if clicked_image_id in df.index:
                    clicked_class = df.loc[clicked_image_id]['class_name']
                    
                    # Toggle selection
                    if current_selected_class == clicked_class:
                        new_selected_class = None  # Deselect
                    else:
                        new_selected_class = clicked_class  # Select
                        
    elif trigger_id == 'histogram' and histogram_click:
        # Handle histogram click
        if 'points' in histogram_click and histogram_click['points']:
            # Extract the full class name from customdata instead of the truncated x value
            # customdata[0] contains the full class name, while x contains the truncated display name
            point = histogram_click['points'][0]
            if 'customdata' in point and point['customdata']:
                clicked_class = point['customdata'][0]
                if clicked_class:
                    # Toggle selection
                    if current_selected_class == clicked_class:
                        new_selected_class = None  # Deselect
                    else:
                        new_selected_class = clicked_class  # Select
    
    # Build appropriate histogram based on current state
    from src.widgets import histogram as histogram_widget
    
    # Priority: Scatterplot selection > CIR results > Full dataset
    if scatterplot_fig and scatterplot_fig['data']:
        main_trace = scatterplot_fig['data'][0]
        if 'selectedpoints' in main_trace and main_trace['selectedpoints'] and len(main_trace['selectedpoints']) > 0:
            # There's an active scatterplot selection - use it for histogram data (highest priority)
            from src.widgets.scatterplot import get_data_selected_on_scatterplot
            selected_data = get_data_selected_on_scatterplot(scatterplot_fig)
            highlight_classes = [new_selected_class] if new_selected_class else None
            histogram_fig = histogram_widget.draw_histogram(selected_data, highlight_classes)
        elif cir_toggle_state and search_data:
            # No scatterplot selection but CIR is active - show CIR results with class highlighting
            df = Dataset.get()
            topk_ids = search_data.get('topk_ids', [])
            cir_df = df.loc[topk_ids]
            highlight_classes = [new_selected_class] if new_selected_class else None
            histogram_fig = histogram_widget.draw_histogram(cir_df, highlight_classes)
        else:
            # No scatterplot selection and no CIR - show empty histogram (don't default to full dataset)
            histogram_fig = histogram_widget.draw_histogram(None)
    else:
        # Fallback
        histogram_fig = histogram_widget.draw_histogram(None)
    
    return new_selected_class, histogram_fig 