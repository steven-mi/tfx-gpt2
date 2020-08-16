# Train GPT-2 in TFX 
Main work done by [nsheppert/gpt-2](https://github.com/nshepperd/gpt-2)

In this repository, we take the existing code and transform it into a TFX pipeline. The TFX pipeline can then be orchestrated with either Airflow or Kubeflow. Have a look at `examples` for getting to know how to use this module.

## Usage
```
git clone https://github.com/steven-mi/tfx-gpt2.git
pip install tfx_gtp2
```

## TODO
- test kubeflow example
- create component for exporting model to TFServing
- resume training
