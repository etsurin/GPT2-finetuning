# wolf_chatbox

## part1: finetune a dialogpt

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

Where <|endoftext|> is the special token of gpt-2 tokenizer. You can also print a sample of raw_data and then you should know how to process your own data. 

ほかのデータでも上記の形に処理しておけば学習データとして扱える。コード内にサンプルをプリントして見ればわかるはず。

-
The default setting is that downloaded csv data and code files are in the same folder. 

csvファイルとコードが同じフォルダにあることを確認してください。

### environment
TBW
### run the code with multiple hyperparameters
TBW
## part2: get the were-wolf data
TBW
