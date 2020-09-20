import os

from datetime import datetime

from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig

from tfx_gpt2.templates.local_custom_language_pipeline import create_pipeline

model_name = "117M"

text_dir = "./data"
text_token_size = 5000  # https://github.com/rkfg/gpt-2/issues/4

mlflow_tracking_url = "./mlruns"

train_config = {'num_iterations': 2,  # number of iterations
                'batch_size': 1,  # Batch size
                'learning_rate': 0.00002,  # Learning rate for Adam
                'accumulate_gradients': 1,  # Accumulate gradients across N minibatches.
                'memory_saving_gradients': False,  # Use gradient checkpointing to reduce vram usage.
                'only_train_transformer_layers': False,  # Restrict training to the transformer blocks.
                'optimizer': 'adam',  # Optimizer. <adam|sgd>.
                'noise': 0.0,  # Add noise to input training data to regularize against typos.

                'top_k': 40,  # K for top-k sampling.
                'top_p': 0.0,  # P for top-p sampling. Overrides top_k if set > 0.

                'sample_every': 1,  # Generate samples every N steps
                'sample_length': 1023,  # Sample this many tokens
                'sample_num': 1,  # Generate this many samples
                'save_every': 1,  # Write a checkpoint every N steps
                }

output_dir = "./output"

pipeline = create_pipeline(pipeline_name=os.path.basename(__file__),
                           pipeline_root=output_dir,
                           model_name=model_name,
                           text_dir=text_dir,
                           text_token_size=text_token_size,
                           mlflow_tracking_url=mlflow_tracking_url,
                           train_config=train_config,
                           enable_cache=True)

airflow_config = {'schedule_interval': "@once",  # every 30 minutes
                  'start_date': datetime(1998, 2, 23, 8),  # year, month, day, hour
                  }
DAG = AirflowDagRunner(AirflowPipelineConfig(airflow_config)).run(pipeline)
