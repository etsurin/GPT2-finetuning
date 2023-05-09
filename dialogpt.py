from pathlib import Path
from turtle import setup
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import random
import evaluate
from data import *
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
tokenizer.padding_side = 'left'
model.resize_token_embeddings(len(tokenizer))
device = 'cuda' if torch.cuda.is_available else 'cpu'
train_data, val_data = train_test_split(all_data,test_size = 0.1)
trainset = ConversationDataset(tokenizer,  train_data, 512)
valset = ConversationDataset(tokenizer,  val_data, 512)
optimizer = optim.Adam(model.parameters(), lr = 5e-5)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle=False, num_workers=0)
n_epoches = 10
model = model.to(device)
accumulate_step = 4
print('***********do training***********')
batch_counter = 0
step_counter = 0
log_step = 200
log_file = open('training_log.txt','w')
print_loss = True
bleu = evaluate.load("bleu")
max_bleu = 0.0
PATH = './dialogpts/bestmodel.pth'
for epoch in range(n_epoches):
    train_loss = 0.0
    for id,item in tqdm(enumerate(trainloader)):
        model.train()
        data,label,_,_ = item
        data = data.to(device)
        label = label.to(device)
        outputs = model(data, labels=label)
        loss = outputs[0]
        loss = loss / accumulate_step
        batch_counter += 1
        if batch_counter % accumulate_step == (accumulate_step - 1):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step_counter += 1
            print_loss = True
        train_loss = train_loss + loss.item()
        if (step_counter % log_step == log_step - 1) and print_loss:
                print('[%d, %d] loss: %.3f' % (epoch + 1, step_counter + 1, train_loss / log_step))
                log_file.write('[%d, %d] loss: %.3f' % (epoch + 1, step_counter + 1, train_loss / log_step)+'\n')
                train_loss = 0.0
                print_loss = False
    
    val_loss = 0.0
    val_counter = 0
    pred_list = []
    with torch.no_grad():
        for id,item in tqdm(enumerate(valloader)):
            model.eval()
            data,label,gensrc,gentgt = item
            data = data.to(device)
            label = label.to(device)
            gensrc = gensrc.to(device)
            outputs = model(data, labels=label)
            pred_tokens = model.generate(gensrc,max_length = 600, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature = 0.8)
            pred_tokens = pred_tokens[:,gensrc.shape[-1]:]
            pred_strings = tokenizer.batch_decode(pred_tokens,skip_special_tokens=True)
            pred_list = pred_list + pred_strings if len(pred_list) else pred_strings
            loss = outputs[0]
            val_loss = val_loss + loss.item()
            val_counter += 1
        val_loss = val_loss / val_counter
        results = bleu.compute(predictions=pred_list, references=valset.gentgt)
    print('epoch{}, valid_loss:{}, bleu:{}'.format(epoch+1,val_loss,100*results['bleu']))
    if results['bleu'] > max_bleu:
        max_bleu = results['bleu']
        torch.save(model.state_dict(), PATH)
        sampled_preds = np.random.choice(pred_list,10,replace = False)
        print('*******best model updated, check outputs*******')
        print(pred_list[:10])
  
para = torch.load(PATH)
model.load_state_dict(para)
model = model.to(device)

def chat(tokenizer,model):
    with torch.no_grad():

        # Let's chat for 5 lines
        for step in range(5):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt').to(device)

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            bot_input_ids = bot_input_ids
            # generated a response while limiting the total chat history to 1000 tokens, 
            chat_history_ids = model.generate(bot_input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature = 0.8)

            # pretty print last ouput tokens from bot
            print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

chat(tokenizer,model)