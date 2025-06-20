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
    selected_image_ids: list,
    base_save_dir: Optional[str] = None
) -> Tuple[list, Optional[Dict]]:
    """
    Perform CIR queries for enhanced prompts with optional saliency generation.
    
    Args:
        temp_image_path: Path to the temporary uploaded image
        enhanced_prompts: List of enhanced prompt strings
        top_n: Number of top results to return per prompt
        selected_image_ids: List of IDs of the selected target images
        base_save_dir: Optional base directory for saving saliency data
    
    Returns:
        Tuple of (all_prompt_results, combined_saliency_data)
    """
    # Prepare results containers
    all_prompt_results: list = []  # retrieval results for every prompt
    prompt_saliency_records: Dict[str, Dict] = {}  # prompt -> saliency metadata

    # Determine base directory where enhanced prompt saliency will be stored
    if base_save_dir is not None:
        root_save_dir = Path(base_save_dir)
    else:
        # Fallback – create a new root directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_save_dir = config.SALIENCY_OUTPUT_DIR / f"enhanced_prompts_{timestamp}"

    root_save_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(enhanced_prompts):
        # 1. Run retrieval for this prompt
        prompt_results = cir_systems.cir_system_searle.query(temp_image_path, prompt, top_n)
        all_prompt_results.append(prompt_results)

        # 2. Generate saliency (full, same settings as initial query) if enabled
        if (config.SALIENCY_ENABLED and
            cir_systems.saliency_manager is not None and
            hasattr(cir_systems.cir_system_searle, 'phi')):
            try:
                prompt_saliency_data = cir_systems.saliency_manager.query_with_saliency(
                    cir_system=cir_systems.cir_system_searle,
                    reference_image_path=temp_image_path,
                    relative_caption=prompt,
                    top_k=top_n,
                    dataset_path=config.DATASET_ROOT_PATH,
                    generate_reference_saliency=config.SALIENCY_GENERATE_REFERENCE,
                    generate_candidate_saliency=config.SALIENCY_GENERATE_CANDIDATES,
                    max_candidate_saliency=config.SALIENCY_MAX_CANDIDATES,
                    generate_text_attribution=config.SALIENCY_GENERATE_TEXT_ATTRIBUTION
                )

                # 3. Save visualisations for this prompt in its own sub-directory
                safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
                sub_dir = root_save_dir / f"prompt_{i+1}_{safe_prompt}"
                cir_systems.saliency_manager.save_saliency_visualizations(prompt_saliency_data, str(sub_dir))

                # Update saliency data with directory info
                prompt_saliency_data['save_directory'] = str(sub_dir)

                # Store record
                prompt_saliency_records[prompt] = prompt_saliency_data

            except Exception as e:
                print(f"⚠️ Failed to generate saliency for enhanced prompt {i+1}: {e}")

    # Build combined object for downstream use
    combined_saliency_data: Optional[Dict] = None
    if prompt_saliency_records:
        combined_saliency_data = {
            'base_directory': str(root_save_dir),
            'prompt_saliency': prompt_saliency_records
        }

    return all_prompt_results, combined_saliency_data


def get_saliency_status_message(saliency_data: Optional[Dict]) -> str:
    """
    Get a status message describing what saliency maps were generated.
    
    Args:
        saliency_data: Saliency data dictionary or None
    
    Returns:
        Human-readable status message
    """
    # New structure: if saliency_data contains enhanced prompt records
    if saliency_data and 'prompt_saliency' in saliency_data:
        num_prompts = len(saliency_data['prompt_saliency'])
        base_dir = Path(saliency_data.get('base_directory', ''))
        if num_prompts:
            return f"Generated saliency for {num_prompts} enhanced prompt(s)"

    # Original behaviour for single-query saliency
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
        return f"Generated {', '.join(messages)}"
    else:
        return "No saliency maps generated" 