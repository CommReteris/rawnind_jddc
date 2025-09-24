'''Batch inference and testing using clean API.

This module provides programmatic batch inference and model testing
without CLI dependencies. It replaces the legacy test_all_known.py script
with modern interfaces that can be used from other packages.

Key features:
- Batch testing across multiple models and test types
- Clean configuration via dataclasses
- No subprocess calls or CLI parsing
- Integrates with clean model loading and metrics computation
'''

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import torch

# Use clean API imports
from .clean_api import InferenceConfig, load_model_from_checkpoint, compute_image_metrics, create_rgb_denoiser, create_bayer_denoiser
from ..dataset.clean_api import create_test_dataset
from ..dependencies import utilities


@dataclass
class BatchTestConfig:
    '''Configuration for batch model testing.'''
    
    # Model selection
    model_types: List[str] = field(default_factory=lambda: ['denoise', 'dc'])
    model_inputs: List[str] = field(default_factory=lambda: ['bayer', 'prgb', 'proc'])
    banned_models: List[str] = field(default_factory=list)
    allowed_models: List[str] = field(default_factory=list)
    
    # Test selection
    tests: List[str] = field(default_factory=lambda: [
        'test_manproc', 'test_manproc_hq', 'test_manproc_q99', 'test_manproc_q995',
        'test_manproc_gt', 'test_manproc_bostitch', 'test_ext_raw_denoise',
        'test_playraw', 'test_manproc_playraw', 'validate_and_test',
        'test_progressive_rawnind', 'test_progressive_manproc', 'test_progressive_manproc_bostitch'
    ])
    
    # Execution parameters
    use_cpu: bool = False
    models_root_dir: str = '/orb/benoit_phd/models/'
    timeout_per_test: int = 12 * 60 * 60  # 12 hours
    
    # Output
    failed_tests_log: str = 'logs/failed_tests.log'


def run_batch_tests(config: BatchTestConfig) -> Dict[str, Any]:
    '''Run batch tests across multiple models using clean API.
    
    Args:
        config: BatchTestConfig with test parameters
        
    Returns:
        Dict with test results including failed tests and summary statistics
    '''
    failed_tests: List[str] = []
    test_results: Dict[str, Dict[str, Any]] = {}
    
    device = 'cpu' if config.use_cpu else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_type in config.model_types:
        models_root_dpath = os.path.join(config.models_root_dir, f'rawnind_{model_type}')
        models_yaml_fpath = os.path.join('config', f'trained_{model_type}_models.yaml')
        
        logging.info(f'Testing models in {models_yaml_fpath}')
        
        with open(models_yaml_fpath, 'r') as f:
            trained_models = yaml.safe_load(f)
        
        # Shuffle for randomized testing
        try:
            trained_models = utilities.shuffle_dictionary(trained_models)
        except AttributeError:
            logging.warning('Could not shuffle models')
        
        for model_name, model_attrs in trained_models.items():
            # Filter models
            if any(banned in model_name for banned in config.banned_models):
                logging.info(f'Skipping banned model: {model_name}')
                continue
            if config.allowed_models and not any(allowed in model_name for allowed in config.allowed_models):
                logging.info(f'Skipping model not in allowed list: {model_name}')
                continue
            
            model_dpath = os.path.join(models_root_dpath, model_name)
            
            # Determine model input/output type
            in_channels = model_attrs.get('in_channels', 3)
            processed_input = model_attrs.get('processed_input', False)
            
            if in_channels == 3:
                if processed_input and 'proc' in config.model_inputs:
                    model_input = 'proc'
                elif 'prgb' in config.model_inputs:
                    model_input = 'prgb'
                else:
                    logging.info(f'Skipping RGB model with unwanted input type: {model_name}')
                    continue
            elif in_channels == 4 and 'bayer' in config.model_inputs:
                model_input = 'bayer'
            else:
                logging.info(f'Skipping model with unsupported channels: {model_name}')
                continue
            
            # Run tests
            model_test_results = {}
            for test_name in config.tests:
                # Skip incompatible test-model combinations
                if 'progressive_manproc' in test_name and model_type == 'dc':
                    logging.info(f'Skipping progressive_manproc for DC model: {model_name}')
                    continue
                if 'bm3d' in model_name and test_name not in ('test_manproc', 'test_progressive_manproc'):
                    logging.info(f'Skipping BM3D test {test_name}')
                    continue
                if ('manproc_playraw' in test_name or 'manproc_hq' in test_name or 
                    'manproc_q99' in test_name or 'manproc_q995' in test_name or 
                    'manproc_gt' in test_name) and model_type == 'denoise':
                    logging.info(f'Skipping denoise manproc variant {test_name}')
                    continue
                if 'manproc' not in test_name and processed_input:
                    logging.info(f'Skipping non-manproc test {test_name} for proc2proc model')
                    continue
                
                logging.info(f'Testing {model_name} with {test_name}')
                
                # Create test dataset using clean API
                test_config = InferenceConfig(
                    architecture=model_attrs.get('arch', 'unet'),
                    input_channels=in_channels,
                    device=device,
                    test_crop_size=512,  # Default test crop size
                    metrics=['msssim', 'psnr']  # Default metrics
                )
                
                # Load model using clean API
                try:
                    model_instance = load_model_from_checkpoint(model_dpath, device=device)
                    
                    # Run test using clean validation method
                    test_dataloader = create_test_dataset(
                        yaml_paths=[models_yaml_fpath],  # Use model config for test data
                        input_channels=in_channels,
                        crop_size=test_config.test_crop_size
                    )
                    
                    test_results = model_instance.validate(
                        test_dataloader,
                        compute_metrics=test_config.metrics_to_compute,
                        save_outputs=False
                    )
                    
                    model_test_results[test_name] = {
                        'success': True,
                        'metrics': test_results,
                        'duration': 0.0  # Placeholder - implement timing
                    }
                    logging.info(f'{test_name} passed for {model_name}: {test_results}')
                    
                except Exception as e:
                    logging.error(f'{test_name} failed for {model_name}: {e}')
                    model_test_results[test_name] = {
                        'success': False,
                        'error': str(e),
                        'duration': 0.0
                    }
                    failed_tests.append(f'{model_name}:{test_name}')
            
            test_results[model_name] = model_test_results
    
    # Save failed tests log
    if failed_tests:
        os.makedirs(os.path.dirname(config.failed_tests_log), exist_ok=True)
        with open(config.failed_tests_log, 'w') as f:
            f.write('\n'.join(failed_tests))
        logging.warning(f'{len(failed_tests)} tests failed. See {config.failed_tests_log}')
    else:
        logging.info('All tests passed successfully')
    
    return {
        'failed_tests': failed_tests,
        'total_tests': len(config.tests) * len(trained_models),
        'test_results': test_results
    }


if __name__ == '__main__':
    # For programmatic usage, create config with defaults
    config = BatchTestConfig()
    
    # Optional: Override defaults from environment variables or config file
    # For example, load from YAML if needed
    results = run_batch_tests(config)
    print(f'Batch testing complete. Failed: {len(results["failed_tests"])} / {results["total_tests"]}')
