# wolf_chatbox

## part1: finetune a dialogpt
First, it's essential to know how to finetune a dialogue model using own data. 

まずは言語モデルのファインチューニングのしかたを知ることです。

### preparing training data

-
I used scripts of Rick and Morty for the learning data. The flow of finetuning is referred to https://colab.research.google.com/drive/15wa925dj7jvdvrz8_z3vU7btqAFQLVlG, thanks to the authors!

学習データとしてRick and Mortyの脚本を使いました。上記のURLを参照してコードを作成しました。著者たちのシェア、ありがとうございます！

-
For other data, process raw_data into this format:

raw_data is a list consisting of dialogue samples
```
[<sample_1>,<sample_2>,...<sample_n>]
```
for each sample with m turns should be a str format that has the following format:
```
[turn_1]<|endoftext|>[turn_2]<|endoftext|>...[turn_m]<|endoftext|>
```

Where <|endoftext|> is the special token of gpt-2 tokenizer. You can also print a sample of raw_data in the code and then you should know how to process your own data. 

ほかのデータでも上記の形に処理しておけば学習データとして扱えます。よくわからなかったらコード内にサンプルをプリントして見ればわかるはずと思います。

-
The default setting is that downloaded csv data and code files are in the same folder. 

csvファイルとコードが同じフォルダにあることを確認してください。

### environment
run the following command
```
pip install -r requirements.txt
```
上記のコマンドを実行してください。

### run the code with multiple hyperparameters
-
run the following command if you want to run in default hyperparameter settings
```
python dial_main.py
```
you can change hyperparameters by adding some of the followings to the original command
```
python dial_main.py --initial_point <model name> --schedule <optimizer schedule> --wm_ratio <warmup steps's ratio> \
--max_len_dial <padding length> --lr <max learning rate> --test_size <ratio to divide to valid set> \
--train_bz <training batch_size> --acc_step <accumulate step> --epoch <training epoches> --log_step <intervals to print loss> \
--max_len_gen <max length for generating responses> --no_repeat_ngram_size <size to avoid generating repeated ngrams> \
--top_k <parameter of top_k sampling method> --top_p <parameter of top_k sampling method> --temp <generating temperature> 
```

ハイパーパラメータを変えたい場合は、上記ようにいずれかを元のコマンドに付けてください。

-
gpt2 series and dialogpt series (infact they are inherited from gpt2 series) are available

gpt2系とdialogpt系はこのコードでファインチューニングできます。

-
accumulate step is an alternation to multi batch size in limited gpu memory. The actual batch size is acc_step * train_bz. 

for generation hyperparameters, please refer to https://huggingface.co/blog/how-to-generate

勾配累積および生成に関するハイパーパラメータの説明です。

### chat with your pretrained model
-
after training model, you can play with it by running the following command (in default settings)

訓練したモデルを遊びたい場合は、以下のコマンドを実行してください。

```
python dial_main.py --play
```

-
you can change hyperparameters by adding some of the followings to the original command
```
python dial_main.py --play --ptr_model_path <the path of your trained model> --chat_turns <maximum chat turns when running the model> \
--no_repeat_ngram_size <size to avoid generating repeated ngrams> \
--top_k <parameter of top_k sampling method> --top_p <parameter of top_k sampling method> --temp <generating temperature> 
```

ハイパーパラメータを変えたい場合は、上記ようにいずれかを元のコマンドに付けてください。

### gpu requirements and time cost

I ran the code on a gpu with 24GB memory.

for training small models e.g. microsoft/dialogpt-small, the training can be finished in several minutes. 

for training large models e.g. microsoft/dialogpt-large, you should restrict the train_bz to 1 and change acc_step for mini-batch training, the training can be finished in 2 hours. 

Of course there is also microsoft/dialogpt-medium. 

24GBのGPUで十分です。ただし、大サイズのモデルをファインチューニングする場合はtrain_bz = 1に限られています。バッチサイズを大きくしたい場合は勾配累積利用してください。 

小サイズのモデルは数分、大サイズでも2時間くらい完成できます。

中サイズもひとつの選択です。

## part2: get the were-wolf data
TBW
