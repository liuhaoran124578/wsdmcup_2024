## Hardware Specifications
- **CPU Cores**: 128
- **Memory**: 768 GB
- **GPU**: NVIDIA Tesla A100 80G
- **Number of GPUs**: 8
- **OS/Platform**: Linux
## Third-Party Software
- **Python**: 3.10.14
- **PyTorch**: 2.3.1+cu121
- **CUDA**: 12.2
- **cuDNN**: 8.9.2.26



# Explanation of directory tree
```
./model_path # pretrained model path
./src # The complete process of my solution
./data # train data and other data
./model_save # save path for train model
./model_save_or
```

# Model Download Preparation
Download these three models to the model_path folder:<br>
Llama3.1 70b (https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) (rename as llama3.1_70b)<br>
Qwen2.5 72b (https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) (rename as qwen2.5_72b)<br>
Gemma2-9b-it-simpo (https://huggingface.co/princeton-nlp/gemma-2-9b-it-SimPO) (rename as Gemma2_9b)<br>
deepseekr1 70b     (https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) (rename as deepseek_r1)


# train model script
（
The time estimates are based on a single machine with 8 NVIDIA A100 80GB GPUs）
1. **Preprocess Data** 
    ```bash
    cd ./src
    sh run_data.sh
    ```
Convert the data into a format suitable for training.
2. **Post-Pretrain**
    ```bash
    sh run_post_pretrain_ut_three_model.sh
    ```
Use UT data to perform post-pretraining on three models
3. **Fine-Tune Models for 5-Fold Results**
    ```bash
    sh run_pipeline.sh 0
    sh run_pipeline.sh 1
    sh run_pipeline.sh 2
    sh run_pipeline.sh 3
    sh run_pipeline.sh 4
    ```
Regarding distillation, the losses we use are as follows:
```python
loss_fun = nn.CrossEntropyLoss()
divergence_loss_fn = nn.KLDivLoss(reduction='batchmean')
cos_loss_fn = nn.CosineEmbeddingLoss()
outputs = model(batch['input_ids'], use_cache=False) # predict gemma2
logits = outputs.logits
grads = batch['grads']
grads1 = batch['grads'][:, :3] # qwen2 
grads2 = batch['grads'][:, 3:] # llama3
labels = batch['labels']
loss_ce = loss_fun(logits, labels)
loss_grad1 = divergence_loss_fn(
    F.log_softmax(logits / T, dim=1),
    F.softmax(grads1 / T, dim=1)
)
cos_loss1 = cos_loss_fn(F.softmax(grads1 / T, dim=1), F.softmax(logits / T, dim=1),
                        torch.ones(logits.size()[0]).to(logits.device))

loss_grad2 = divergence_loss_fn(
    F.log_softmax(logits / T, dim=1),
    F.softmax(grads2 / T, dim=1)
)
cos_loss2 = cos_loss_fn(F.softmax(grads2 / T, dim=1), F.softmax(logits / T, dim=1),
                        torch.ones(logits.size()[0]).to(logits.device))

loss = (loss_ce + loss_grad1 + cos_loss1 + loss_grad2 + cos_loss2) / 5.
```

Each fold includes training the Llama3.1 and Qwen2.5(deepseekr1) models, predicting to obtain the probability distribution for the training set, and finally fine-tuning the Gemma model.
4. **Merge LoRA and Quantize**
    ```bash
    sh run_final.sh
    ```
Here, the LoRA layers of the 5-fold Gemma models are merged and then quantized to 8-bit.
5. **Predict Test Set**
    ```python
    python predict_test.py
    ```
Once the previous steps are completed, you can directly run this script to make predictions.The final results will be saved in ./sub/submission.csv
If there is a new test set, you can directly replace ./data/lmsys-chatbot-arena/test.csv for prediction.<br>
