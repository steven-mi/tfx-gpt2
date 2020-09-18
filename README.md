# GPT-2 in TFX 
This repository contains code for creating a end-end TFX pipeline for GPT-2. The pipeline contains the needed data preprocessing, exporting data from MongoDB, training with pretrained model and deployment to a MLFlow model registry. In the registry there is a possibility to label all produced models. Thus one can image that a service only gets the production model. All pipelines can be orchestrated with either Airflow, Beam or Kubeflow. Tensorboard is supported and can be used for keeping a track during training. 

**Why TFX?**

- TFX is a open source framework for creating production grade end-end machine learning pipelines.
- It handles a lot of useful things like caching and versioning
- Support for Airflow and Kubeflow

**Why Tensorboard?**

- Tensorflow native visualization tool
- Powerful tool

**Why MLFlow**

- Built-in Model Registy
- Dashboard for comparing models, their performance and produces artifacts

## Getting Started

### Install package
```bash
git clone https://github.com/steven-mi/tfx-gpt2.git
cd tfx-gpt2
pip indstall tfx-gpt2
```

### Run pipeline with Apache Beam

```bash
cd examples
python beam-local-example.py
# open mlflow and tensorboard
mlflow server --backend-store-uri sqlite:///mlflow.d --default-artifact-root ./mlruns
tensorboard --logdir ./outputs
```

### Run with Apache Airflow

```bash
# ... setup airflow
# copy files to dag folder
cp airflow-local-example.py $AIRFLOW_HOME/dags
cp -r data $AIRFLOW_HOME/dags
# start scheduler and executor
airflow scheduler -D
airflow webserver 
# go to localhost:8080 an start the DAG
# outputs are stored in $AIRFLOW_HOME/dags and can be changed in airflow-local-example.py
cd $AIRFLOW_HOME/dags
mlflow server --backend-store-uri sqlite:///mlflow.d --default-artifact-root ./mlruns
tensorboard --logdir ./outputs
```


## List of available models

Look at the [Paper](https://openai.com/blog/better-language-models/) for performance details. Choose the model depending on your resources. Insert the model name as string value for `model_name` in your pipeline.

| Model Name        | Layers | Comments                                                     |
| ----------------- | ------ | ------------------------------------------------------------ |
| `117M`<br/>`124M` | 12     | Could be trained on a CPU or a single for a reasonable amount of time |
| `345M`<br/>`355M` | 24     | Could be trained on a single GPU for a reasonable amount of time |
| `774M`            | 36     | Not recommended. If you have enough ressources for training this model, then `1558M` should be fine too. |
| `1558M`           | 48     | Largest model trained. Known for super human performance (see: https://openai.com/blog/better-language-models/) |


## Training recommendations
- GPT-2 models' robustness and worst case behaviors are not well-understood. As with any machine-learned model, carefully evaluate GPT-2 for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset our GPT-2 models were trained on contains many texts with biases and factual inaccuracies, and thus GPT-2 models are likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination. Our models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.

From [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple): The method GPT-2 uses to generate text is slightly different than those like other packages like textgenrnn (specifically, generating the full text sequence purely in the GPU and decoding it later), which cannot easily be fixed without hacking the underlying model code. As a result:
- In general, GPT-2 is better at maintaining context over its entire generation length, making it good for generating conversational text. The text is also generally gramatically correct, with proper capitalization and few typoes.
- The original GPT-2 model was trained on a very large variety of sources, allowing the model to incorporate idioms not seen in the input text.
- GPT-2 can only generate a maximum of 1024 tokens per request (about 3-4 paragraphs of English text).
- GPT-2 cannot stop early upon reaching a specific end token. (workaround: pass the truncate parameter to a generate function to only collect text until a specified end token. You may want to reduce length appropriately.)
- Higher temperatures work better (e.g. 0.7 - 1.0) to generate more interesting text, while other frameworks work better between 0.2 - 0.5.
- When finetuning GPT-2, it has no sense of the beginning or end of a document within a larger text. You'll need to use a bespoke character sequence to indicate the beginning and end of a document. Then while generating, you can specify a prefix targeting the beginning token sequences, and a truncate targeting the end token sequence. You can also set include_prefix=False to discard the prefix token while generating (e.g. if it's something unwanted like <|startoftext|>).
- If you pass a single-column .csv file to finetune(), it will automatically parse the CSV into a format ideal for training with GPT-2 (including prepending <|startoftext|> and suffixing <|endoftext|> to every text document, so the truncate tricks above are helpful when generating output). This is necessary to handle both quotes and newlines in each text document correctly.
- GPT-2 allows you to generate texts in parallel by setting a batch_size that is divisible into nsamples, resulting in much faster generation. Works very well with a GPU (can set batch_size up to 20 on Colaboratory's K80)!
- Due to GPT-2's architecture, it scales up nicely with more powerful GPUs. For the 124M model, if you want to train for longer periods of time, GCP's P100 GPU is about 3x faster than a K80/T4 for only 3x the price, making it price-comparable (the V100 is about 1.5x faster than the P100 but about 2x the price). The P100 uses 100% of the GPU even with batch_size=1, and about 88% of the V100 GPU.
- If you have a partially-trained GPT-2 model and want to continue finetuning it, you can set overwrite=True to finetune, which will continue training and remove the previous iteration of the model without creating a duplicate copy. This can be especially useful for transfer learning (e.g. heavily finetune GPT-2 on one dataset, then finetune on other dataset to get a "merging" of both datasets).
- The 774M "large" model may support finetuning because it will cause modern GPUs to go out-of-memory (you may get lucky if you use a P100 GPU on Colaboratory). However, you can still generate from the default pretrained model using gpt2.load_gpt2(sess, model_name='774M') and gpt2.generate(sess, model_name='774M').
- The 1558M "extra large", true model, may not work out-of-the-box with the GPU included with the Colaboratory Notebook. More testing is needed to identify optimial configurations for it.
