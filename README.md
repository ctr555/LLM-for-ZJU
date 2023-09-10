# LLM-for-ZJU
主要参考这个仓库 https://github.com/ymcui/Chinese-LLaMA-Alpaca

### 配置环境注意：

https://github.com/huggingface/peft里面下载，然后传输到服务器

```python
scp -r -P 10022 ./peft-main.zip ctr@10.72.74.122:/home/ctr/LLM/
pip install peft-main.zip
pip uninstall nvidia_cublas_cu11
pip install protobuf==3.19.0
```

### 1、向122服务器传输下面的文件

连接服务器

```python
ssh -p 10022 ctr@10.72.74.122
caitianrun123
```

![截屏2023-09-07 16.39.43.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/40eb6d45-b87c-4e7f-bbad-47bbdb26919c/69e0bb09-0a51-4325-8991-86ad5f9b4bb5/%E6%88%AA%E5%B1%8F2023-09-07_16.39.43.png)

然后按照下面命令进行解压缩

```python
unzip chinese_alpaca_plus_lora_7b.zip -d chinese_alpaca_plus_lora_7b
unzip chinese_llama_plus_lora_7b.zip -d chinese_llama_plus_lora_7b
```

最后llama_models下面会得到这些东西

![截屏2023-09-07 16.42.30.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/40eb6d45-b87c-4e7f-bbad-47bbdb26919c/bb358f8f-1c2a-43c8-851d-79db42f1aa49/%E6%88%AA%E5%B1%8F2023-09-07_16.42.30.png)

### 2、**下载hugging face里面的transformer**

122服务器不行git clone需要自己传输

```python
scp -r -P 10022 ./transformers ctr@10.72.74.109:/home/lj/LLM/Chinese-LLaMA-Alpaca-main
```

进入到这个路径下面

```python
cd transformers/src/transformers/models/llama/
```

然后执行这个命令

（注意llama_models是自己的路径）

```python
python convert_llama_weights_to_hf.py --input_dir /home/lj/LLM//llama_models/ --model_size 7B --output_dir /home/lj/LLM//llama_models/7B_hf/
```

等待一下，然后会出现下面的东西

![截屏2023-09-07 17.57.49.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/40eb6d45-b87c-4e7f-bbad-47bbdb26919c/aa58d07d-6279-47b2-981e-af213e1e148b/%E6%88%AA%E5%B1%8F2023-09-07_17.57.49.png)

### 3、**合并LoRA权重，生成全量模型权重**

```python
python scripts/merge_llama_with_chinese_lora.py \
--base_model /home/ctr/LLM/llama_models/7B_hf/ \
--lora_model /home/ctr/LLM/llama_models/chinese_alpaca_plus_lora_7b,/home/ctr/LLM/llama_models/chinese_llama_plus_lora_7b \
--output_dir /home/ctr/LLM/llama_models/7B_full_model \
--output_type huggingface
```

这里直接会报错，所以执行下面命令

```python
vim scripts/merge_llama_with_chinese_lora.py
：262
```

注释262-264行

### 4、**使用GPU或者CPU推理**

```python
python scripts/inference/inference_hf.py \
--base_model /home/ctr/LLM/llama_models/7B_full_model \
--with_prompt \
--interactive
```

### 5、WebUI 下载text-generation-webui项目

```python
git clone https://github.com/oobabooga/text-generation-webui
pip install -r requirements.txt
```

### 6、将转换后的7B-hf放到models下面

### 7、替换models下面的文件

```python
cp llama_models/chinese_alpaca_plus_lora_7b/tokenizer.model text-generation-webui-main/models/llama-7B-hf/
cp llama_models/chinese_alpaca_plus_lora_7b/special_tokens_map.json text-generation-webui-main/models/llama-7B-hf/
cp llama_models/chinese_alpaca_plus_lora_7b/tokenizer_config.json text-generation-webui-main/models/llama-7B-hf/
```

### 8、修改代码

/modules/LoRA.py

```python
shared.model.resize_token_embeddings(len(shared.tokenizer))

linux正则匹配
?shared.model = Peft
?len(lora_names)
```

### 9、运行脚本

```python
python server.py --listen-host 0.0.0.0 --listen-port 7860 --model models/llama-7B-hf/ --lora loras/chinese-alpaca-lora-7b/ --cpu
python server.py --model models/llama-7B-hf/ --lora loras/chinese-alpaca-lora-7b/ --cpu
```

然后vim server.py，launch()里面的share=True

然后打开网页的地址

```python
http://127.0.0.1:7860
http://localhost:7860
发现不可以
```

![截屏2023-09-08 18.28.15.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/40eb6d45-b87c-4e7f-bbad-47bbdb26919c/c44a46c7-9475-4a8d-9e3d-e9bc3422b0b4/%E6%88%AA%E5%B1%8F2023-09-08_18.28.15.png)

解决方案：https://github.com/gradio-app/gradio/issues/4548

https://github.com/THUDM/VisualGLM-6B/issues/135

```python
手动下载文件，移动这个这个位置
/home/ctr/miniconda3/envs/LLM/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2

cp frpc_linux_amd64 /home/ctr/miniconda3/envs/LLM/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2
chmod -x /home/ctr/miniconda3/envs/LLM/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2

```

```python
查看监听的端口命令
netstat -tuln

https://blog.csdn.net/weixin_44409833/article/details/127177310

端口映射到本地
ssh -L7860:127.0.0.1:7860 -p 10022 ctr@10.72.74.122

这样就可以打开啦！
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/40eb6d45-b87c-4e7f-bbad-47bbdb26919c/f1aafae2-bec5-4ead-b98d-db1c0f49cbd1/Untitled.png)

### 10、llama 微调还没弄好
