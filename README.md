# wolf_chatbox

## part1: finetune a dialogpt
First, it's essential to know how to finetune a dialogue model using own data. 

まずは言語モデルのファインチューニングのしかたを知ること。

### preparing training data

-
I used scripts of Rick and Morty for the learning data. The flow of finetuning is referred to https://colab.research.google.com/drive/15wa925dj7jvdvrz8_z3vU7btqAFQLVlG, thanks to the authors!

学習データとしてRick and Mortyの脚本を使った。上記のURLを参照してコードを作成した。著者たちのシェア、ありがとうございます！

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

ほかのデータでも上記の形に処理しておけば学習データとして扱える。コード内にサンプルをプリントして見ればわかるはず。

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

ハイパーパラメータを変えたい場合は、上記のいずれかを元のコマンドに付けてください。

accumulate step is an alternation to multi batch size in limited gpu memory. The actual batch size is acc_step * train_bz. 

for generation hyperparameters, please refer to https://huggingface.co/blog/how-to-generate


### chat with your pretrained model
after training model, you can play with it by running the following command (in default settings)
```
python dial_main.py --play
```

you can change hyperparameters by adding some of the followings to the original command
```
python dial_main.py --play --chat_turns <maximum chat turns when running the model> --no_repeat_ngram_size <size to avoid generating repeated ngrams> \
--top_k <parameter of top_k sampling method> --top_p <parameter of top_k sampling method> --temp <generating temperature> 
```


## part2: get the were-wolf data
TBW
