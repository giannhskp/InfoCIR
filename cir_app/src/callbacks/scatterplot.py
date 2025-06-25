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
    cir_toggle_state, selectedData, relayoutData, clickData, deselect_clicks,
    gallery_clicks, wordcloud_click, selected_scatterplot_class, prompt_selection, viz_mode, viz_selected_ids,
    scatterplot_fig, search_data, selected_gallery_image_ids, selected_image_data, 
    enhanced_prompts_data, selected_histogram_class
):
    """
    Unified controller that handles all scatterplot updates from different triggers.
    Priority order: deselect > CIR toggle > selection > relayout > click > others
    """
    import dash
    from dash import callback_context
    import plotly.graph_objects as go
    from src import config
    from src.Dataset import Dataset
    from src.widgets import scatterplot
    
    # Get the trigger that caused this callback
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    trigger_value = ctx.triggered[0]['value']
    
    # print(f"Scatterplot controller triggered by: {trigger_id} (value: {trigger_value})")
    
    # DEBUG: Log current figure state
    if scatterplot_fig and 'data' in scatterplot_fig:
        trace_names = [trace.get('name', 'unnamed') for trace in scatterplot_fig['data']]
        # print(f"DEBUG: Input figure has {len(scatterplot_fig['data'])} traces: {trace_names}")
        
        # Check for selections in input figure
        layout_selections = scatterplot_fig.get('layout', {}).get('selections', [])
        main_trace_selectedpoints = None
        if scatterplot_fig['data']:
            main_trace_selectedpoints = scatterplot_fig['data'][0].get('selectedpoints', [])
        # print(f"DEBUG: Input layout.selections: {len(layout_selections) if layout_selections else 0} rectangles")
        # print(f"DEBUG: Input main trace selectedpoints: {len(main_trace_selectedpoints) if main_trace_selectedpoints else 0} points")
        
        # Check dragmode
        dragmode = scatterplot_fig.get('layout', {}).get('dragmode', 'unknown')
        # print(f"DEBUG: Input dragmode: {dragmode}")
        
        # Check revision counters
        sel_rev = scatterplot_fig.get('layout', {}).get('selectionrevision', 'none')
        ui_rev = scatterplot_fig.get('layout', {}).get('uirevision', 'none')
        # print(f"DEBUG: Input selectionrevision: {sel_rev}, uirevision: {ui_rev}")
    # else:
        # print("DEBUG: Input figure is None or has no data")
    
    # DEBUG: Log other relevant states
    # print(f"DEBUG: cir_toggle_state: {cir_toggle_state}")
    # print(f"DEBUG: search_data is None: {search_data is None}")
    # print(f"DEBUG: selectedData: {selectedData}")
    # print(f"DEBUG: relayoutData: {relayoutData}")

    # -------------------------------------------------------------------------
    # 1. DESELECT BUTTON - Clear selections but preserve CIR traces if active
    # -------------------------------------------------------------------------
    if trigger_id == 'deselect-button':
        # print("Handling deselect button")
        # print(f"DEBUG: DESELECT - deselect_clicks value: {deselect_clicks}")
        
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # print(f"DEBUG: DESELECT - Before clearing, layout.selections: {len(new_fig.get('layout', {}).get('selections', []))}")
        # print(f"DEBUG: DESELECT - Before clearing, main trace selectedpoints: {len(new_fig['data'][0].get('selectedpoints', []) if new_fig['data'] else [])}")
        
        # Clear selections and overlays
        new_fig['layout']['selections'] = []  # CHANGED: Use empty list instead of None
        new_fig['layout']['images'] = []
        
        # Remove selection traces but keep CIR traces
        old_trace_count = len(new_fig['data'])
        new_fig['data'] = [
            trace for trace in new_fig['data']
            if trace.get('name') not in ['Selected Images', 'Selected Image']
        ]
        # print(f"DEBUG: DESELECT - Removed selection traces: {old_trace_count} -> {len(new_fig['data'])}")
        
        # Clear selectedpoints from main trace
        if new_fig['data'] and 'selectedpoints' in new_fig['data'][0]:
            old_selectedpoints = len(new_fig['data'][0]['selectedpoints'])
            new_fig['data'][0]['selectedpoints'] = []
            # print(f"DEBUG: DESELECT - Cleared selectedpoints: {old_selectedpoints} -> 0")
        # else:
            # print("DEBUG: DESELECT - No selectedpoints to clear")
            
        # Always reset main trace colors to clear any class highlighting
        from src.widgets.scatterplot import _set_marker_colors
        _set_marker_colors(new_fig['data'][0], config.SCATTERPLOT_COLOR)
        # print("DEBUG: DESELECT - Reset main trace colors")
        
        # Clear class highlighting legend - remove selected class trace only, don't add legend
        old_trace_count = len(new_fig['data'])
        new_fig['data'] = [trace for trace in new_fig['data'] if trace.get('name') != 'selected class']
        # print(f"DEBUG: DESELECT - Removed selected class traces: {old_trace_count} -> {len(new_fig['data'])}")
        
        # Ensure we have exactly one "image embedding" legend trace
        # Only count traces that actually appear in legend (not the main data trace)
        legend_traces = [trace for i, trace in enumerate(new_fig['data']) 
                        if i > 0 and trace.get('name') == 'image embedding']
        has_legend = len(legend_traces) > 0
        # print(f"DEBUG: DESELECT - Legend traces found: {len(legend_traces)}, has_legend: {has_legend}")
        
        if not has_legend:
            # Add legend trace if missing
            legend_trace = go.Scatter(
                x=[None], y=[None], mode="markers", name='image embedding',
                marker=dict(size=7, color="blue", symbol='circle')
            )
            new_fig['data'].append(legend_trace.to_plotly_json())
            # print("DEBUG: DESELECT - Added missing legend trace")
        
        # END PATCH
        
        # FORCE CLEAR: Toggle dragmode momentarily
        prev_dragmode = new_fig['layout'].get('dragmode', 'select')
        # print(f"DEBUG: DESELECT - Previous dragmode: {prev_dragmode}")
        if prev_dragmode in ('select', 'lasso'):
            new_fig['layout']['dragmode'] = 'zoom'
            # print("DEBUG: DESELECT - Changed dragmode to zoom")
        
        # NEW PATCH: bump selectionrevision so Plotly definitely discards the selection overlay
        sel_rev = new_fig['layout'].get('selectionrevision', 0)
        try:
            sel_rev = int(sel_rev)
        except Exception:
            sel_rev = 0
        new_fig['layout']['selectionrevision'] = sel_rev + 1
        
        # FORCE FULL REDRAW: bump uirevision as well so Plotly redraws completely
        new_fig['layout']['uirevision'] = sel_rev + 1  # use same counter
        
        # print(f"DEBUG: DESELECT - Set selectionrevision: {sel_rev} -> {sel_rev + 1}")
        # print(f"DEBUG: DESELECT - Set uirevision: {sel_rev + 1}")
        
        # Final debug check
        final_trace_names = [trace.get('name', 'unnamed') for trace in new_fig['data']]
        final_selections = len(new_fig.get('layout', {}).get('selections', []))
        final_selectedpoints = len(new_fig['data'][0].get('selectedpoints', []) if new_fig['data'] else [])
        final_dragmode = new_fig['layout'].get('dragmode', 'unknown')
        
        # print(f"DEBUG: DESELECT - Final figure: {len(new_fig['data'])} traces: {final_trace_names}")
        # print(f"DEBUG: DESELECT - Final selections: {final_selections}, selectedpoints: {final_selectedpoints}")
        # print(f"DEBUG: DESELECT - Final dragmode: {final_dragmode}")
        # print("DEBUG: DESELECT - Returning modified figure")

        return new_fig
    
    # -------------------------------------------------------------------------
    # 2. CIR TOGGLE STATE CHANGE - Add/remove result traces
    # -------------------------------------------------------------------------
    elif trigger_id == 'cir-toggle-state':
        # print(f"Handling CIR toggle: {cir_toggle_state}")
        # print(f"DEBUG: CIR-TOGGLE - search_data is None: {search_data is None}")
        # if search_data:
            # print(f"DEBUG: CIR-TOGGLE - search_data keys: {list(search_data.keys())}")
        
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # print(f"DEBUG: CIR-TOGGLE - Input figure selections: {len(new_fig.get('layout', {}).get('selections', []))}")
        # print(f"DEBUG: CIR-TOGGLE - Input figure selectedpoints: {len(new_fig['data'][0].get('selectedpoints', []) if new_fig['data'] else [])}")
        
        if not cir_toggle_state:
            # print("DEBUG: CIR-TOGGLE - Hiding CIR results")
            # Hide CIR results - remove all CIR traces but preserve main data and legend traces
            new_fig['layout']['images'] = []
            from src.widgets.scatterplot import _set_marker_colors
            _set_marker_colors(new_fig['data'][0], config.SCATTERPLOT_COLOR)
            
            # Keep main data trace and legend traces, remove only CIR traces
            preserved_traces = []
            has_legend = False
            
            old_trace_count = len(new_fig['data'])
            for i, trace in enumerate(new_fig['data']):
                trace_name = trace.get('name', '')
                if trace_name not in ['Top-K', 'Top-1', 'Query', 'Final Query']:
                    preserved_traces.append(trace)
                    # Only count legend traces, not the main data trace (index 0)
                    if trace_name == 'image embedding' and i > 0:
                        has_legend = True
                # else:
                    # print(f"DEBUG: CIR-TOGGLE - Removing CIR trace: {trace_name}")
            
            # Ensure we have exactly one legend trace
            if not has_legend:
                legend_trace = go.Scatter(
                    x=[None], y=[None], mode="markers", name='image embedding',
                    marker=dict(size=7, color="blue", symbol='circle')
                )
                preserved_traces.append(legend_trace.to_plotly_json())
                # print("DEBUG: CIR-TOGGLE - Added missing legend trace")
            
            new_fig['data'] = preserved_traces
            # BEGIN PATCH: ensure any active box/lasso selections are fully cleared when hiding CIR traces
            # Clear Plotly selection rectangles (if any)
            selections_before = len(new_fig.get('layout', {}).get('selections', []))
            if 'selections' in new_fig.get('layout', {}):
                new_fig['layout']['selections'] = []
            # Clear any selected points stored in the main trace to avoid persisting highlights
            selectedpoints_before = len(new_fig['data'][0].get('selectedpoints', []) if new_fig['data'] else [])
            if new_fig['data'] and 'selectedpoints' in new_fig['data'][0]:
                new_fig['data'][0]['selectedpoints'] = []
            # END PATCH
            # print(f"DEBUG: CIR-TOGGLE - Cleared selections: {selections_before} -> 0")
            # print(f"DEBUG: CIR-TOGGLE - Cleared selectedpoints: {selectedpoints_before} -> 0")
            # print(f"DEBUG: CIR-TOGGLE - After hiding CIR, preserved {len(new_fig['data'])} traces (was {old_trace_count})")
            
        elif search_data:
            # print("DEBUG: CIR-TOGGLE - Showing CIR results")
            # Show CIR results - add result traces
            df = Dataset.get()
            topk_ids = search_data.get('topk_ids', [])
            top1_id = search_data.get('top1_id', None)
            
            # print(f"DEBUG: CIR-TOGGLE - Adding CIR traces for {len(topk_ids)} results")
            
            # Clear existing CIR traces first
            old_trace_count = len(new_fig['data'])
            new_fig['data'] = [
                trace for trace in new_fig['data']
                if trace.get('name') not in ['Top-K', 'Top-1', 'Query', 'Final Query']
            ]
            new_fig['layout']['images'] = []
            # print(f"DEBUG: CIR-TOGGLE - Cleared existing CIR traces: {old_trace_count} -> {len(new_fig['data'])}")
            
            # Get coordinates from main trace
            main_trace = new_fig['data'][0]
            xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
            
            # Determine projection type for query positioning
            axis_title = new_fig['layout']['xaxis']['title']['text']
            
            if axis_title == 'umap_x':
                xq, yq = search_data.get('umap_x_query'), search_data.get('umap_y_query')
                xfq, yfq = search_data.get('umap_x_final_query'), search_data.get('umap_y_final_query')
            else:
                xq, yq = None, None
                xfq, yfq = None, None
            
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
                # print(f"DEBUG: CIR-TOGGLE - Added Top-K trace with {len(xk)} points, color={config.TOP_K_COLOR}")
                
            if x1:
                trace_1 = go.Scatter(
                    x=x1, y=y1, mode='markers', 
                    marker=dict(color=config.TOP_1_COLOR, size=9, opacity=1.0), 
                    name='Top-1',
                    showlegend=True,
                    visible=True
                )
                new_fig['data'].append(trace_1.to_plotly_json())
                #   print(f"DEBUG: CIR-TOGGLE - Added Top-1 trace with {len(x1)} points, color={config.TOP_1_COLOR}")
                
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
                # print(f"DEBUG: CIR-TOGGLE - Added Query trace at ({xq}, {yq}), color={config.QUERY_COLOR}")
                
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
                # print(f"DEBUG: CIR-TOGGLE - Added Final Query trace at ({xfq}, {yfq}), color={config.FINAL_QUERY_COLOR}")
            
            # print(f"DEBUG: CIR-TOGGLE - Final figure has {len(new_fig['data'])} traces total")
            
            # Debug: Print trace names to verify they're there
            trace_names = [trace.get('name', 'unnamed') for trace in new_fig['data']]
            # print(f"DEBUG: CIR-TOGGLE - Trace names: {trace_names}")
        
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
            # print(f"DEBUG: CIR-TOGGLE - Added missing 'image embedding' legend trace")
        
        # Apply zoom-responsive sizing to all traces including newly added CIR traces
        if 'layout' in new_fig and new_fig['layout'].get('xaxis', {}).get('range'):
            zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
            # print(f"DEBUG: CIR-TOGGLE - Applying zoom factor {zoom_factor} to CIR traces")
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
        
        # Final debug check to ensure CIR traces are properly added
        final_trace_names = [trace.get('name', 'unnamed') for trace in new_fig['data']]
        final_selections = len(new_fig.get('layout', {}).get('selections', []))
        final_selectedpoints = len(new_fig['data'][0].get('selectedpoints', []) if new_fig['data'] else [])
        # print(f"DEBUG: CIR-TOGGLE - Final trace count: {len(new_fig['data'])}, names: {final_trace_names}")
        # print(f"DEBUG: CIR-TOGGLE - Final selections: {final_selections}, selectedpoints: {final_selectedpoints}")
        # print("DEBUG: CIR-TOGGLE - Returning modified figure")
        
        return new_fig
    
    # -------------------------------------------------------------------------
    # 4. ZOOM/RELAYOUT - Add thumbnail images when zoomed in and apply zoom-responsive sizing
    # -------------------------------------------------------------------------
    elif trigger_id == 'scatterplot' and relayoutData:
        print("Handling scatterplot zoom/relayout")
        # print(f"DEBUG: RELAYOUT - relayoutData = {relayoutData}")
        
        # BEGIN PATCH: guard against stale selection rectangles fired AFTER we pressed Deselect.
        # If the relayout event only carries a 'selections' payload but the main trace currently
        # contains **no** selected points, treat it as stale and discard the rectangle.
        if 'selections' in relayoutData:
            main_trace = scatterplot_fig['data'][0] if scatterplot_fig and scatterplot_fig.get('data') else None
            has_active_pts = bool(main_trace and main_trace.get('selectedpoints'))
            # print(f"DEBUG: RELAYOUT - Found selections in relayoutData, has_active_pts: {has_active_pts}")
            # print(f"DEBUG: RELAYOUT - Main trace selectedpoints: {len(main_trace.get('selectedpoints', []) if main_trace else [])}")
            if not has_active_pts:
                # print("DEBUG: RELAYOUT - Discarding stale selection rectangle from relayoutData (no selected points)")
                # Remove selections from relayoutData and ensure they don't get re-applied
                original_relayout = relayoutData.copy()
                relayoutData = {k: v for k, v in relayoutData.items() if k != 'selections'}
                # print(f"DEBUG: RELAYOUT - Modified relayoutData: {original_relayout} -> {relayoutData}")
                # Also strip any lingering selections in the figure we clone below
                if 'layout' in scatterplot_fig and 'selections' in scatterplot_fig['layout']:
                    scatterplot_fig['layout'].pop('selections', None)
                    # print("DEBUG: RELAYOUT - Removed lingering selections from input figure")
        # END PATCH
        
        # Skip dragmode changes
        if len(relayoutData) == 1 and 'dragmode' in relayoutData:
            # print("DEBUG: RELAYOUT - Skipping dragmode-only change")
            return dash.no_update
        
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # Check if this is an axis reset (autorange) or explicit zoom
        is_axis_reset = ('xaxis.autorange' in relayoutData or 'yaxis.autorange' in relayoutData)
        has_explicit_range = ('xaxis.range[0]' in relayoutData or 'yaxis.range[0]' in relayoutData)
        
        # print(f"DEBUG: RELAYOUT - is_axis_reset: {is_axis_reset}, has_explicit_range: {has_explicit_range}")
        
        if is_axis_reset:
            # print('DEBUG: RELAYOUT - Handling axis reset - restoring original marker sizes')
            # Axis reset: apply zoom factor of 1.0 (original sizes)
            zoom_factor = 1.0
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
            # Clear thumbnail images when resetting
            new_fig['layout']['images'] = []
            
        elif has_explicit_range:
            # print('DEBUG: RELAYOUT - Adding thumbnail overlays and applying zoom-responsive sizing')
            # Apply zoom-responsive marker sizing
            zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
            # print(f"DEBUG: RELAYOUT - Calculated zoom factor: {zoom_factor}")
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
            
            # Add thumbnail images with zoom-responsive sizing
            new_fig = scatterplot.add_images_to_scatterplot(new_fig, zoom_factor)
        else:
            # Other relayout changes - apply zoom-responsive sizing if we have ranges
            # print(f"DEBUG: RELAYOUT - Other relayout change: {relayoutData}")
            # Don't block the update, but apply zoom-responsive sizing if possible
            if 'layout' in new_fig and new_fig['layout'].get('xaxis', {}).get('range'):
                zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
                # print(f"DEBUG: RELAYOUT - Applying zoom factor {zoom_factor} to other relayout change")
                new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
        
        # Final debug check
        final_selections = len(new_fig.get('layout', {}).get('selections', []))
        final_selectedpoints = len(new_fig['data'][0].get('selectedpoints', []) if new_fig['data'] else [])
        # print(f"DEBUG: RELAYOUT - Final selections: {final_selections}, selectedpoints: {final_selectedpoints}")
        # print("DEBUG: RELAYOUT - Returning modified figure")
        
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
        
        # BEGIN PATCH: ensure previous selection rectangles/highlights are cleared before applying new selection
        if 'selections' in new_fig.get('layout', {}):
            new_fig['layout']['selections'] = []
        if new_fig['data'] and 'selectedpoints' in new_fig['data'][0]:
            new_fig['data'][0]['selectedpoints'] = []
        # END PATCH
        
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
                            name='Query',
                            showlegend=True,
                            visible=True
                        )
                        new_fig['data'].append(trace_q.to_plotly_json())
                        
                    if xfq is not None:
                        trace_fq = go.Scatter(
                            x=[xfq], y=[yfq], mode='markers', 
                            marker=dict(color=config.FINAL_QUERY_COLOR, size=10, symbol='diamond', opacity=1.0), 
                            name='Final Query',
                            showlegend=True,
                            visible=True
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
            # print("DEBUG: Class selection while CIR is active - preserving CIR traces")
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
    # 8. PROMPT SELECTION - Handle enhanced prompt selection/deselection
    # -------------------------------------------------------------------------
    elif trigger_id == 'prompt-selection':
        print(f"Handling prompt selection: {prompt_selection}")
        # print(f"DEBUG: PROMPT-SELECTION - prompt_selection value: {prompt_selection}")
        
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # print(f"DEBUG: PROMPT-SELECTION - Input figure selections: {len(new_fig.get('layout', {}).get('selections', []))}")
        # print(f"DEBUG: PROMPT-SELECTION - Input figure selectedpoints: {len(new_fig['data'][0].get('selectedpoints', []) if new_fig['data'] else [])}")
        
        if prompt_selection == -1:
            # Prompt deselected - clear any selections that might be lingering
            # print("DEBUG: PROMPT-SELECTION - Prompt deselected, clearing selections")
            
            # Clear selections and overlays
            selections_before = len(new_fig.get('layout', {}).get('selections', []))
            new_fig['layout']['selections'] = []
            new_fig['layout']['images'] = []
            # print(f"DEBUG: PROMPT-SELECTION - Cleared layout selections: {selections_before} -> 0")
            
            # Clear selectedpoints from main trace
            selectedpoints_before = len(new_fig['data'][0].get('selectedpoints', []) if new_fig['data'] else [])
            if new_fig['data'] and 'selectedpoints' in new_fig['data'][0]:
                new_fig['data'][0]['selectedpoints'] = []
            # print(f"DEBUG: PROMPT-SELECTION - Cleared selectedpoints: {selectedpoints_before} -> 0")
            
            # Reset main trace colors to clear any class highlighting
            from src.widgets.scatterplot import _set_marker_colors
            _set_marker_colors(new_fig['data'][0], config.SCATTERPLOT_COLOR)
            # print("DEBUG: PROMPT-SELECTION - Reset main trace colors")
            
            # Remove selection traces
            old_trace_count = len(new_fig['data'])
            new_fig['data'] = [
                trace for trace in new_fig['data']
                if trace.get('name') not in ['Selected Images', 'Selected Image', 'selected class']
            ]
            # print(f"DEBUG: PROMPT-SELECTION - Removed selection traces: {old_trace_count} -> {len(new_fig['data'])}")
            
            # Ensure we have exactly one "image embedding" legend trace
            legend_traces = [trace for i, trace in enumerate(new_fig['data']) 
                            if i > 0 and trace.get('name') == 'image embedding']
            has_legend = len(legend_traces) > 0
            if not has_legend:
                legend_trace = go.Scatter(
                    x=[None], y=[None], mode="markers", name='image embedding',
                    marker=dict(size=7, color="blue", symbol='circle')
                )
                new_fig['data'].append(legend_trace.to_plotly_json())
                # print("DEBUG: PROMPT-SELECTION - Added missing legend trace")
            
            # Force clear with revision bump
            sel_rev = new_fig['layout'].get('selectionrevision', 0)
            try:
                sel_rev = int(sel_rev)
            except Exception:
                sel_rev = 0
            new_fig['layout']['selectionrevision'] = sel_rev + 1
            new_fig['layout']['uirevision'] = sel_rev + 1
            # print(f"DEBUG: PROMPT-SELECTION - Bumped revisions to: {sel_rev + 1}")
            
        else:
            # Enhanced prompt functionality - show specific prompt results
            # print(f"DEBUG: PROMPT-SELECTION - Specific prompt selected: {prompt_selection}")
            
            if not cir_toggle_state or not enhanced_prompts_data:
                # print("DEBUG: PROMPT-SELECTION -
                return dash.no_update
            
            # Clear existing CIR traces
            new_fig['data'] = [
                trace for trace in new_fig['data']
                if trace.get('name') not in ['Top-K', 'Top-1', 'Query', 'Final Query']
            ]
            
            # Determine which results to show
            if prompt_selection is not None and prompt_selection >= 0:
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
                    # print(f"DEBUG: PROMPT-SELECTION - Using enhanced prompt {prompt_selection} with {len(topk_ids)} results")
                else:
                    # print("DEBUG: PROMPT-SELECTION - Invalid prompt selection index")
                    return dash.no_update
            else:
                # print("DEBUG: PROMPT-SELECTION - Invalid prompt selection value")
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
                        # print(f"DEBUG: prompt-selection falling back to original coords: x={xfq}, y={yfq}")
                else:
                    # Use original Final Query coordinates
                    xfq, yfq = search_data.get('umap_x_final_query'), search_data.get('umap_y_final_query')
                    # print(f"DEBUG: prompt-selection using original coords: x={xfq}, y={yfq}")
                    
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
                # print(f"Applying zoom factor {zoom_factor} to enhanced prompt traces")
                new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
        
        # Final debug check
        final_trace_names = [trace.get('name', 'unnamed') for trace in new_fig['data']]
        final_selections = len(new_fig.get('layout', {}).get('selections', []))
        final_selectedpoints = len(new_fig['data'][0].get('selectedpoints', []) if new_fig['data'] else [])
        # print(f"DEBUG: PROMPT-SELECTION - Final trace count: {len(new_fig['data'])}, names: {final_trace_names}")
        # print(f"DEBUG: PROMPT-SELECTION - Final selections: {final_selections}, selectedpoints: {final_selectedpoints}")
        # print("DEBUG: PROMPT-SELECTION - Returning modified figure")
        
        return new_fig
    
    # -------------------------------------------------------------------------
    # 9. VISUALIZATION MODE - Handle viz mode selection changes
    # -------------------------------------------------------------------------
    elif trigger_id == 'viz-selected-ids':
        print(f"Handling viz selection: {viz_selected_ids}")
        # print(f"DEBUG: viz_mode = {viz_mode}")
        # print(f"DEBUG: cir_toggle_state = {cir_toggle_state}")
        # print(f"DEBUG: search_data is None = {search_data is None}")
        # print(f"DEBUG: prompt_selection = {prompt_selection}")
        
        import copy
        new_fig = copy.deepcopy(scatterplot_fig)
        
        # CRITICAL: Check if CIR is active and we need to preserve/add CIR traces
        # This handles the race condition where viz-selected-ids and cir-toggle-state 
        # are both triggered simultaneously by the CIR search callback
        # This must happen BEFORE the viz_mode check to handle the race condition
        if cir_toggle_state and search_data:
            # print("DEBUG: viz-selected-ids detected CIR is active, preserving/adding CIR traces")
            
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
                    # print(f"DEBUG: viz-selected-ids using enhanced prompt {prompt_selection} with {len(topk_ids)} results")
                else:
                    # Fallback to original search data
                    topk_ids = search_data.get('topk_ids', [])
                    top1_id = search_data.get('top1_id', None)
                    # print(f"DEBUG: viz-selected-ids falling back to original search data")
            else:
                # Use original search data
                topk_ids = search_data.get('topk_ids', [])
                top1_id = search_data.get('top1_id', None)
                # print(f"DEBUG: viz-selected-ids using original search data")
            
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
                xq, yq = None, None
                xfq, yfq = None, None
            
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
            
            # print(f"DEBUG: viz-selected-ids re-added CIR traces, now have {len(new_fig['data'])} traces")
            
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
                # print(f"DEBUG: viz-selected-ids added missing 'image embedding' legend trace")
        else:
            # Remove existing Selected Images trace but preserve other traces
            new_fig['data'] = [tr for tr in new_fig['data'] if tr.get('name') != 'Selected Images']
        
        # print(f"DEBUG: viz-selected-ids preserving {len(new_fig['data'])} traces")
        
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
                # print(f"DEBUG: Added Selected Images trace with {len(sel_x)} points")
        
        # Apply zoom-responsive sizing to all traces including visualization mode traces
        if 'layout' in new_fig and new_fig['layout'].get('xaxis', {}).get('range'):
            zoom_factor = scatterplot.calculate_zoom_factor(new_fig['layout'])
            # print(f"Applying zoom factor {zoom_factor} to visualization mode traces")
            new_fig = scatterplot.apply_zoom_responsive_sizing(new_fig, zoom_factor)
         
        # print(f"DEBUG: viz-selected-ids returning figure with {len(new_fig['data'])} traces")
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