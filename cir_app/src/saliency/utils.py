#!/usr/bin/env python3
"""
Utility functions for integrating saliency with CIR systems.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from src import config
from src.shared import cir_systems


def perform_cir_with_saliency(
    temp_image_path: str,
    text_prompt: str,
    top_n: int,
    selected_model: str = "searle"
) -> Tuple[Any, Optional[Dict]]:
    """
    Perform CIR query with optional saliency generation.
    
    Args:
        temp_image_path: Path to the temporary uploaded image
        text_prompt: Text describing the desired modification
        top_n: Number of top results to return
        selected_model: Model to use ("searle" or "freedom")
    
    Returns:
        Tuple of (cir_results, saliency_data)
        - cir_results: Standard CIR results from the chosen system
        - saliency_data: Dictionary with saliency maps or None if disabled
    """
    # Perform standard CIR query
    with cir_systems.lock:
        if selected_model == "freedom":
            cir_results = cir_systems.cir_system_freedom.query(temp_image_path, text_prompt, top_n)
        else:  # default to searle
            cir_results = cir_systems.cir_system_searle.query(temp_image_path, text_prompt, top_n)
    
    # Generate saliency if enabled and using SEARLE (which has phi network)
    saliency_data = None
    if (config.SALIENCY_ENABLED and 
        selected_model.lower() == "searle" and 
        cir_systems.saliency_manager is not None and
        hasattr(cir_systems.cir_system_searle, 'phi') and
        cir_systems.cir_system_searle.phi is not None):
        
        try:
            print("🔍 Generating saliency maps...")
            
            saliency_data = cir_systems.saliency_manager.query_with_saliency(
                cir_system=cir_systems.cir_system_searle,
                reference_image_path=temp_image_path,
                relative_caption=text_prompt,
                top_k=top_n,
                dataset_path=config.DATASET_ROOT_PATH,
                generate_reference_saliency=config.SALIENCY_GENERATE_REFERENCE,
                generate_candidate_saliency=config.SALIENCY_GENERATE_CANDIDATES,
                max_candidate_saliency=config.SALIENCY_MAX_CANDIDATES,
                generate_text_attribution=config.SALIENCY_GENERATE_TEXT_ATTRIBUTION
            )
            
            # Save saliency visualizations with timestamp
            if saliency_data and ('reference' in saliency_data.get('saliency_maps', {}) or 
                                 'candidates' in saliency_data.get('saliency_maps', {})):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_prompt = "".join(c for c in text_prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
                save_dir = config.SALIENCY_OUTPUT_DIR / f"query_{timestamp}_{safe_prompt}"
                
                cir_systems.saliency_manager.save_saliency_visualizations(
                    saliency_data, str(save_dir)
                )
                
                # Add save path to saliency data for reference
                saliency_data['save_directory'] = str(save_dir)
                
                print(f"✅ Saliency maps saved to: {save_dir}")
            
        except Exception as e:
            print(f"⚠️ Failed to generate saliency maps: {e}")
            # Don't let saliency errors break the main CIR functionality
            import traceback
            traceback.print_exc()
    
    return cir_results, saliency_data


def perform_enhanced_prompt_cir_with_saliency(
    temp_image_path: str,
    enhanced_prompts: list,
    top_n: int,
    selected_image_id: str
) -> Tuple[list, Optional[Dict]]:
    """
    Perform CIR queries for enhanced prompts with optional saliency generation.
    
    Args:
        temp_image_path: Path to the temporary uploaded image
        enhanced_prompts: List of enhanced prompt strings
        top_n: Number of top results to return per prompt
        selected_image_id: ID of the selected target image
    
    Returns:
        Tuple of (all_prompt_results, combined_saliency_data)
    """
    all_prompt_results = []
    combined_saliency_data = None
    
    # Process each enhanced prompt
    for i, prompt in enumerate(enhanced_prompts):
        prompt_results = cir_systems.cir_system_searle.query(temp_image_path, prompt, top_n)
        all_prompt_results.append(prompt_results)
        
        # Generate saliency for the first few prompts if enabled
        saliency_limit = config.SALIENCY_MAX_CANDIDATES if config.SALIENCY_MAX_CANDIDATES is not None else 3
        if (config.SALIENCY_ENABLED and 
            i < saliency_limit and  # Limit prompts to avoid too much computation
            cir_systems.saliency_manager is not None and
            hasattr(cir_systems.cir_system_searle, 'phi')):
            
            try:
                prompt_saliency_data = cir_systems.saliency_manager.query_with_saliency(
                    cir_system=cir_systems.cir_system_searle,
                    reference_image_path=temp_image_path,
                    relative_caption=prompt,
                    top_k=top_n,
                    dataset_path=config.DATASET_ROOT_PATH,
                    generate_reference_saliency=False,  # Skip reference for enhanced prompts
                    generate_candidate_saliency=True,
                    max_candidate_saliency=1,  # Only generate for top candidate
                    generate_text_attribution=False
                )
                
                # Store the first prompt's saliency data
                if combined_saliency_data is None and prompt_saliency_data:
                    combined_saliency_data = prompt_saliency_data
                    
            except Exception as e:
                print(f"⚠️ Failed to generate saliency for enhanced prompt {i+1}: {e}")
    
    # Save combined saliency data if available
    if combined_saliency_data:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = config.SALIENCY_OUTPUT_DIR / f"enhanced_prompts_{timestamp}"
            
            cir_systems.saliency_manager.save_saliency_visualizations(
                combined_saliency_data, str(save_dir)
            )
            
            combined_saliency_data['save_directory'] = str(save_dir)
            print(f"✅ Enhanced prompt saliency maps saved to: {save_dir}")
            
        except Exception as e:
            print(f"⚠️ Failed to save enhanced prompt saliency: {e}")
    
    return all_prompt_results, combined_saliency_data


def get_saliency_status_message(saliency_data: Optional[Dict]) -> str:
    """
    Get a status message describing what saliency maps were generated.
    
    Args:
        saliency_data: Saliency data dictionary or None
    
    Returns:
        Human-readable status message
    """
    if not saliency_data:
        if config.SALIENCY_ENABLED:
            # More specific error checking
            if cir_systems.saliency_manager is None:
                return "Saliency generation skipped (saliency manager not initialized)"
            elif not hasattr(cir_systems.cir_system_searle, 'phi'):
                return "Saliency generation skipped (φ network not found in CIR system)"
            elif cir_systems.cir_system_searle.phi is None:
                return "Saliency generation skipped (φ network is None)"
            else:
                return "Saliency generation skipped (unknown reason - check logs for details)"
        else:
            return "Saliency generation disabled"
    
    saliency_maps = saliency_data.get('saliency_maps', {})
    messages = []
    
    if 'reference' in saliency_maps:
        messages.append("reference image saliency")
    
    if 'candidates' in saliency_maps:
        num_candidates = len(saliency_maps['candidates'])
        messages.append(f"{num_candidates} candidate saliency map(s)")
    
    if messages:
        save_dir = saliency_data.get('save_directory', 'unknown location')
        return f"Generated {', '.join(messages)} → saved to {Path(save_dir).name}"
    else:
        return "No saliency maps generated" 