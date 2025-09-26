"""Operation resolver for converting simple names to full specifications."""

import re
from typing import Dict, List, Optional, Any, Tuple
import logging
from .operation_spec import OperationSpec, TensorType
from .operation_registry import OPERATION_REGISTRY

logger = logging.getLogger(__name__)


class OperationResolver:
    """Resolves operation names and configurations from various formats.
    
    This class handles conversion from simple operation names, aliases,
    and shorthand notations to full operation specifications.
    """
    
    def __init__(self, registry=None):
        """Initialize the resolver.
        
        Args:
            registry: Operation registry to use
        """
        self.registry = registry or OPERATION_REGISTRY
        
        # Define common aliases
        self.aliases = {
            "load": "raw_loader",
            "wb": "white_balance",
            "demosaic": "demosaic",
            "color": "color_transform",
            "encode": "encoder",
            "decode": "decoder",
            "save": "save_hdr",
            "to_rggb": "bayer_to_rggb",
            "from_rggb": "rggb_to_bayer",
        }
        
        # Define preset pipelines
        self.presets = {
            "raw_to_rgb": ["raw_loader", "white_balance", "demosaic", "color_transform"],
            "raw_to_rggb": ["raw_loader", "white_balance", "bayer_to_rggb"],
            "raw_to_hdr": ["raw_loader", "white_balance", "demosaic", "color_transform", "save_hdr"],
            "encode_raw": ["raw_loader", "white_balance", "bayer_to_rggb", "encoder"],
            "decode_to_rgb": ["decoder", "color_transform"],
        }
    
    def resolve_operation(self, 
                         operation: str, 
                         config: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Resolve an operation name and configuration.
        
        Args:
            operation: Operation name, alias, or shorthand
            config: Optional configuration
            
        Returns:
            Tuple of (resolved_name, merged_config)
            
        Raises:
            ValueError: If operation cannot be resolved
        """
        # Check if it's a preset pipeline
        if operation in self.presets:
            raise ValueError(
                f"'{operation}' is a preset pipeline, not a single operation. "
                f"Use resolve_pipeline() instead."
            )
        
        # Resolve aliases
        resolved_name = self.aliases.get(operation, operation)
        
        # Check if operation exists
        spec = self.registry.get_spec(resolved_name)
        if not spec:
            # Try fuzzy matching
            matches = self._fuzzy_match(operation)
            if matches:
                resolved_name = matches[0]
                spec = self.registry.get_spec(resolved_name)
                logger.info(f"Resolved '{operation}' to '{resolved_name}'")
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        # Merge configuration with defaults
        merged_config = spec.get_effective_config(config)
        
        return resolved_name, merged_config
    
    def resolve_pipeline(self, 
                        pipeline_spec: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Resolve a pipeline specification string.
        
        Args:
            pipeline_spec: Pipeline specification (e.g., "load -> wb -> demosaic -> save")
            
        Returns:
            List of (operation_name, config) tuples
        """
        # Check if it's a preset
        if pipeline_spec in self.presets:
            operations = self.presets[pipeline_spec]
            return [(op, {}) for op in operations]
        
        # Parse pipeline string
        operations = []
        
        # Split on arrows or pipes
        parts = re.split(r'\s*(?:->|=>|\|)\s*', pipeline_spec)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check for configuration in parentheses
            match = re.match(r'(\w+)(?:\((.*?)\))?', part)
            if match:
                op_name = match.group(1)
                config_str = match.group(2)
                
                # Parse configuration
                config = {}
                if config_str:
                    config = self._parse_config_string(config_str)
                
                # Resolve operation
                resolved_name, merged_config = self.resolve_operation(op_name, config)
                operations.append((resolved_name, merged_config))
            else:
                # Simple operation name
                resolved_name, merged_config = self.resolve_operation(part)
                operations.append((resolved_name, merged_config))
        
        return operations
    
    def _parse_config_string(self, config_str: str) -> Dict[str, Any]:
        """Parse a configuration string.
        
        Args:
            config_str: Configuration string (e.g., "method=bilinear, scale=2")
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Split on commas
        parts = config_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to parse value type
                if value.lower() in ('true', 'false'):
                    config[key] = value.lower() == 'true'
                elif value.isdigit():
                    config[key] = int(value)
                elif self._is_float(value):
                    config[key] = float(value)
                else:
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    config[key] = value
        
        return config
    
    def _is_float(self, value: str) -> bool:
        """Check if a string represents a float.
        
        Args:
            value: String to check
            
        Returns:
            True if string is a float
        """
        try:
            float(value)
            return '.' in value or 'e' in value.lower()
        except ValueError:
            return False
    
    def _fuzzy_match(self, query: str, threshold: float = 0.7) -> List[str]:
        """Find operations that fuzzy match the query.
        
        Args:
            query: Search query
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of matching operation names
        """
        matches = []
        query_lower = query.lower()
        
        for name in self.registry._specs.keys():
            name_lower = name.lower()
            
            # Check substring match
            if query_lower in name_lower or name_lower in query_lower:
                matches.append(name)
                continue
            
            # Check prefix match
            if name_lower.startswith(query_lower) or query_lower.startswith(name_lower):
                matches.append(name)
                continue
            
            # Simple similarity score
            similarity = self._simple_similarity(query_lower, name_lower)
            if similarity >= threshold:
                matches.append(name)
        
        return matches
    
    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple similarity between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        # Levenshtein distance-based similarity
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # Count common characters
        common = sum(1 for c in s1 if c in s2)
        max_len = max(len(s1), len(s2))
        
        return common / max_len
    
    def suggest_operations(self, 
                          context: Optional[str] = None,
                          input_type: Optional[TensorType] = None) -> List[str]:
        """Suggest operations based on context.
        
        Args:
            context: Context or description of what to do
            input_type: Current tensor type
            
        Returns:
            List of suggested operation names
        """
        suggestions = []
        
        if input_type:
            # Get operations that accept this input type
            compatible = self.registry.list_operations(input_type=input_type)
            suggestions.extend(compatible[:5])  # Limit to top 5
        
        if context:
            # Search for operations matching context
            context_lower = context.lower()
            
            for name, spec in self.registry._specs.items():
                if any(word in spec.description.lower() for word in context_lower.split()):
                    if name not in suggestions:
                        suggestions.append(name)
        
        return suggestions[:10]  # Limit total suggestions
    
    def validate_pipeline_string(self, pipeline_spec: str) -> List[str]:
        """Validate a pipeline specification string.
        
        Args:
            pipeline_spec: Pipeline specification string
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            operations = self.resolve_pipeline(pipeline_spec)
            
            # Extract operation names
            op_names = [op[0] for op in operations]
            
            # Validate using registry
            pipeline_errors = self.registry.validate_pipeline(op_names)
            errors.extend(pipeline_errors)
            
        except Exception as e:
            errors.append(f"Failed to parse pipeline: {str(e)}")
        
        return errors
    
    def get_pipeline_info(self, pipeline_spec: str) -> Dict[str, Any]:
        """Get detailed information about a pipeline.
        
        Args:
            pipeline_spec: Pipeline specification string
            
        Returns:
            Dictionary with pipeline information
        """
        operations = self.resolve_pipeline(pipeline_spec)
        
        info = {
            "operations": [],
            "total_operations": len(operations),
            "input_type": None,
            "output_type": None,
            "metadata_required": set(),
            "metadata_generated": set(),
        }
        
        for i, (op_name, config) in enumerate(operations):
            spec = self.registry.get_spec(op_name)
            
            op_info = {
                "index": i,
                "name": op_name,
                "category": spec.category,
                "description": spec.description,
                "config": config,
                "input_types": [t.value for t in spec.input_types],
                "output_types": [t.value for t in spec.output_types],
            }
            
            info["operations"].append(op_info)
            
            # Track first input and last output
            if i == 0 and spec.input_types:
                info["input_type"] = spec.input_types[0].value
            if i == len(operations) - 1 and spec.output_types:
                info["output_type"] = spec.output_types[0].value
            
            # Track metadata
            info["metadata_required"].update(spec.metadata_fields_required)
            info["metadata_generated"].update(spec.metadata_fields_generated)
        
        # Convert sets to lists for JSON serialization
        info["metadata_required"] = list(info["metadata_required"])
        info["metadata_generated"] = list(info["metadata_generated"])
        
        return info