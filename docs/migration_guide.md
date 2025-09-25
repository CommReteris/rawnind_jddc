# Migration Guide: Legacy Dataset Factory to Clean API

## Key Changes

1. **Registry Pattern**: Dataset types are now registered via decorators instead of explicit factory methods
2. **Type Safety**: Stronger validation of configuration parameters
3. **Simplified Interface**: Single factory functions replace multiple builder classes

## Migration Steps

### 1. Update Configuration

```python
# Legacy
config = {
    'dataset_type': 'train',
    'pre_cropped': False,
    'input_channels': 3
}

# New
from rawnind.dataset import DatasetConfig
config = DatasetConfig(
    dataset_type='train',
    pre_cropped=False,
    input_channels=3,
    training_mode=True
)
```

### 2. Dataset Creation

```python
# Legacy
from legacy_rawds import TrainingDatasetBuilder

dataset = TrainingDatasetBuilder(config).build()

# New
from rawnind.dataset import create_training_dataset

dataset = create_training_dataset(config, data_paths)
```

### 3. Custom Dataset Implementation

```python
# Legacy
class CustomDataset(TrainingDataset):
    def __init__(self, config):
        super().__init__(config)
        
# New
@DatasetRegistry.register(trainable=True, pre_cropped=False)
class CustomDataset(ConfigurableDataset):
    def __init__(self, config: DatasetConfig, data_paths: Dict[str, Any]):
        super().__init__(config, data_paths)
```

## Key Benefits

- Stronger type checking and validation
- Centralized configuration management
- Easier implementation of custom dataset types
- Better error messages for invalid configurations