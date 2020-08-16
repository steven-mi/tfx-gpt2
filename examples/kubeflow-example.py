import os

from datetime import datetime

from tfx_gpt2 import create_pipeline

from kfp import onprem
from tfx.orchestration.kubeflow import kubeflow_dag_runner

model_name = "117M"

text_path = "/PATH/TO/DATASET"

train_config = {'num_iterations': 100000,  # number of iterations
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

output_dir = "/PATH/TO/OUTPUT/DIR"

pipeline = create_pipeline(pipeline_name=os.path.basename(__file__),
                           pipeline_root=output_dir,
                           model_name=model_name,
                           text_path=text_path,
                           train_config=train_config)

airflow_config = {'schedule_interval': "@once",  # every 30 minutes
                  'start_date': datetime(1998, 2, 23, 8),  # year, month, day, hour
                  }

# Metadata config. The defaults works work with the installation of
# KF Pipelines using Kubeflow. If installing KF Pipelines using the
# lightweight deployment option, you may need to override the defaults.
metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

# This pipeline automatically injects the Kubeflow TFX image if the
# environment variable 'KUBEFLOW_TFX_IMAGE' is defined. Currently, the tfx
# cli tool exports the environment variable to pass to the pipelines.
tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)

# This sample assumes a persistent volume (PV) is mounted as follows. To use
# InfraValidator with PVC, its access mode should be ReadWriteMany.
_persistent_volume_claim = 'my-pvc'
_persistent_volume = 'my-pv'
_persistent_volume_mount = '/mnt'

runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config=metadata_config,
    # Specify custom docker image to use.
    tfx_image=tfx_image,
    pipeline_operator_funcs=(
        # If running on K8s Engine (GKE) on Google Cloud Platform (GCP),
        # kubeflow_dag_runner.get_default_pipeline_operator_funcs() provides
        # default configurations specifically for GKE on GCP, such as secrets.
        [
            onprem.mount_pvc(_persistent_volume_claim, _persistent_volume,
                             _persistent_volume_mount)
        ]))

kubeflow_dag_runner.KubeflowDagRunner(config=runner_config)