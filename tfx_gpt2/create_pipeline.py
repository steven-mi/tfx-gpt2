# TODO
import os

from tfx.orchestration import metadata
from tfx.orchestration import pipeline

from tfx_gpt2.components.create_components import create_components


def create_pipeline(pipeline_name, pipeline_root, model_name, text_path, train_config, encoding='utf-8', combine=50000,
                    enable_cache=False):
    default_train_config = {'num_iterations': 100000,  # number of iterations
                            'batch_size': 1,  # Batch size
                            'learning_rate': 0.00002,  # Learning rate for Adam
                            'accumulate_gradients': 1,  # Accumulate gradients across N minibatches.
                            'memory_saving_gradients': False,  # Use gradient checkpointing to reduce vram usage.
                            'only_train_transformer_layers': False,  # Restrict training to the transformer blocks.
                            'optimizer': 'adam',  # Optimizer. <adam|sgd>.
                            'noise': 0.0,  # Add noise to input training data to regularize against typos.

                            'top_k': 40,  # K for top-k sampling.
                            'top_p': 0.0,  # P for top-p sampling. Overrides top_k if set > 0.

                            'sample_every': 100,  # Generate samples every N steps
                            'sample_length': 1023,  # Sample this many tokens
                            'sample_num': 1,  # Generate this many samples
                            'save_every': 1000,  # Write a checkpoint every N steps
                            }
    for key, value in train_config.items():
        default_train_config[key] = value

    components = create_components(model_name=model_name,
                                   text_path=text_path,
                                   train_config=default_train_config,
                                   encoding=encoding,
                                   combine=combine)

    pipeline_root = os.path.join(pipeline_root, 'pipelines', pipeline_name)
    metadata_path = os.path.join(pipeline_root, 'metadata', pipeline_name,
                                 'metadata.db')

    tfx_pipeline = pipeline.Pipeline(pipeline_name=pipeline_name,
                                     pipeline_root=pipeline_root,
                                     components=components,
                                     enable_cache=enable_cache,
                                     metadata_connection_config=metadata.sqlite_metadata_connection_config(
                                         metadata_path))
    return tfx_pipeline
