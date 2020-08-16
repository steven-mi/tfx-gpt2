import os

from datetime import datetime

from tfx_gpt2 import create_pipeline

from kfp import onprem
from tfx.orchestration.kubeflow import kubeflow_dag_runner

persistent_volume_claim = 'tfx-pvc'
persistent_volume = 'tfx-pv'
persistent_volume_mount = '/mnt'

model_name = "117M"

text_path = os.path.join(persistent_volume_mount, "data")

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

pipeline = create_pipeline(pipeline_name=os.path.basename(__file__),
                           pipeline_root=persistent_volume_mount,
                           model_name=model_name,
                           text_path=text_path,
                           train_config=train_config)

runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    # Metadata config. The defaults works work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    kubeflow_metadata_config=kubeflow_dag_runner.get_default_kubeflow_metadata_config(),
    pipeline_operator_funcs=(
        # If running on K8s Engine (GKE) on Google Cloud Platform (GCP),
        # kubeflow_dag_runner.get_default_pipeline_operator_funcs() provides
        # default configurations specifically for GKE on GCP, such as secrets.
        [
            onprem.mount_pvc(persistent_volume_claim, persistent_volume,
                             persistent_volume_mount)
        ]))
kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(pipeline)
