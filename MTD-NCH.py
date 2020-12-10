import json
import torch
from transformers import GPT2Tokenizer
from transformers import GPT2DoubleHeadsModel
from MTDNN import MTDNN
from tqdm import trange, tqdm
from keras_preprocessing import sequence
import pandas as pd
import Utils
import pickle
import os
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

SPECIAL_TOKENS = ['<pad>', '<eos>', '<rstokn>', '<bos>', '<question>', '<commonsensetask>', '<cose>', '<openbook>']
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'pad_token': '<pad>', 'eos_token': '<eos>',
                         'additional_special_tokens': ['<rstokn>', '<question>', '<reply>', '<commonsensetask>', '<cose>', '<openbook>']
                         }


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logs_dir_tensorboard = "runs2nomcs/" + (str(current_time) + "morecheckpoint-melco-update-ros")
writer = SummaryWriter(logs_dir_tensorboard)

device = 'cuda:5'
def data_preprocess():
    final_data = []
    questions = []
    choices = []
    label = []
    facts = []

    file_name = 'data/OpenBookFacts/train_complete.jsonl'
    for line in open(file_name, 'r') :
        data = (json.loads(line))

        questions.append(data['question']['stem'])
        choices.append([data['question']['choices'][0]['text'], data['question']['choices'][1]['text'],
                       data['question']['choices'][2]['text'], data['question']['choices'][3]['text']])
        if data['answerKey'] == 'A' :
            answer = 0
        elif data['answerKey'] == 'B' :
            answer = 1
        elif data['answerKey'] == 'C' :
            answer = 2
        else:
            answer = 3
        label.append(answer)
        facts.append(data['fact1'])

    openBook_Data = [["openBook"],  questions, choices, label, facts]
    final_data.append(openBook_Data)

    file_name = 'data/CoS-E/cose_train_data.csv'
    data = pd.read_csv(file_name)
    final_data.append([["CoS-E"], data])

    file_name_1 = 'data/commonsense/subtaskB_data_all-2.csv'
    file_name_2 = 'data/commonsense/subtaskC-alldata.csv'
    data1 = pd.read_csv(file_name_1)
    data2 = pd.read_csv(file_name_2)
    data = data1.merge(data2, on='FalseSent').dropna()
    final_data.append([["commonsense"], data]) # leave the last 500

    return final_data

def convert_to_tokens(input, tokenizer):
    if isinstance(input, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
    elif isinstance(input, list):
        return [
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(val))
            if not isinstance(val, int) else val
            for val in input
        ]
    elif isinstance(input, pd.Series):
        input = input.tolist()
        return [
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(val))
            if not isinstance(val, int) else val
            for val in input
        ]
    else:
        import sys
        print("Conversion Error")
        sys.exit()

def padding_falsesent_choices(datas, tokenizer):

    pad = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[0])]
    eos = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[1])]
    rstokn = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[2])]
    bos = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[3])]
    questons = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[4])]
    commonsensetask = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[5])]
    COSE = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[6])]
    openBook = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[7])]
    choice_padding = -1

    input_ids = []
    lm_labels = []
    token_type_ids = []
    mc_token_ids = []
    mc_labels = []

    max_length = 128

    for data in datas:
        if data[0] == ["openBook"]:

            for question, choices, labels, facts in zip( data[1], data[2], data[3], data[4]):

                # /mc_labels = []

                question, choices, facts = convert_to_tokens(question, tokenizer), convert_to_tokens(choices, tokenizer), convert_to_tokens(facts, tokenizer)
                input1 = [bos + openBook + rstokn + question + rstokn + choices[0] + rstokn + choices[1] + rstokn + choices[2] + rstokn + choices[3] + eos]
                input2 = [bos + openBook + rstokn + question + rstokn + facts + eos]

                mc_token_ids.append(len(input1[0]))
                mc_token_ids.append(len(input2[0]))

                input1 = sequence.pad_sequences(input1, maxlen=max_length, padding='post', value=pad)
                input_ids.append(input1[0])

                fakechoice = sequence.pad_sequences([[-1]], maxlen=max_length, padding='post', value=choice_padding)
                lm_labels.append(fakechoice[0])

                tt_id1 = [
                    (len(openBook) + 1) * rstokn + (len(question) + 1) * questons + (len(choices[0]) + 1) * rstokn +
                    (len(choices[1]) + 1) * questons + (len(choices[2]) + 1) * rstokn + (len(choices[3]) + 2) * questons
                ]
                tt_id1 = sequence.pad_sequences(tt_id1, maxlen=max_length, padding='post', value=pad)
                token_type_ids.append(tt_id1[0])

                input2 = sequence.pad_sequences(input2, maxlen=max_length, padding='post', value=pad)
                input_ids.append(input2[0])

                choice = [[-1] * (len(openBook) + 2) + [-1] * len(question) + [-1] + facts + eos]
                choice = sequence.pad_sequences(choice, maxlen=max_length, padding='post', value=choice_padding)
                lm_labels.append(choice[0])

                tt_id2 = [(len(openBook) + 1) * rstokn + (len(question) + 1) * questons + (len(choices[labels]) + 2) * rstokn]
                tt_id2 = sequence.pad_sequences(tt_id2, maxlen=max_length, padding='post', value=pad)
                token_type_ids.append(tt_id2[0])

                mc_labels.append(labels)
                mc_labels.append(labels)



        elif data[0] == ["CoS-E"]:

            for idx, value in data[1].iterrows():
                value = value[1:]
                value = convert_to_tokens(value, tokenizer)
                input1 = [bos + COSE + rstokn + value[1] + rstokn + value[2] + rstokn + value[3] + rstokn + value[4] +
                          rstokn + value[5] + rstokn + value[6] + eos]
                input2 = [bos + COSE + rstokn + value[1] + rstokn + value[8] + eos]

                mc_token_ids.append(len(input1[0]))
                mc_token_ids.append(len(input2[0]))

                input1 = sequence.pad_sequences(input1, maxlen= max_length, padding='post', value=pad)
                input_ids.append(input1[0])

                fakechoice = sequence.pad_sequences([[-1]], maxlen=max_length, padding='post', value=choice_padding)
                lm_labels.append(fakechoice[0])

                tt_id1 = [(len(COSE) + 1) * rstokn + (len(value[1]) + 1) * questons + (len(value[2]) + 1) * rstokn +
                    (len(value[3]) + 1) * questons + (len(value[4]) + 1) * rstokn + (len(value[5]) + 1) * questons +
                           (len(value[6]) + 2) * rstokn]
                tt_id1 = sequence.pad_sequences(tt_id1, maxlen= max_length, padding='post', value=pad)
                token_type_ids.append(tt_id1[0])

                input2 = sequence.pad_sequences(input2, maxlen=max_length, padding='post', value=pad)
                input_ids.append(input2[0])

                choice = [[-1] * (len(COSE) + 2) + [-1] * len(value[1]) + [-1] + value[8] + eos]
                choice = sequence.pad_sequences(choice, maxlen=max_length, padding='post', value=choice_padding)
                lm_labels.append(choice[0])

                tt_id2 = [(len(COSE) + 1) * rstokn + (len(value[1]) + 1) * questons + (len(value[8]) +2) * rstokn]
                tt_id2 = sequence.pad_sequences(tt_id2, maxlen=max_length, padding='post', value=pad)
                token_type_ids.append(tt_id2[0])

                mc_labels.append(value[7])
                mc_labels.append(value[7])

        elif data[0] == ["commonsense"]:
            for idx, value in data[1].iterrows():
                # call tokenizer
                value = convert_to_tokens(value, tokenizer)
                input1 = [bos + commonsensetask + rstokn + value[1] + rstokn + value[2] + rstokn + value[3] + rstokn + value[4]+ eos]
                ml = input1
                input1 = sequence.pad_sequences(input1, maxlen=max_length, padding='post', value=pad)
                fakechoice = sequence.pad_sequences([[-1]], maxlen=max_length, padding='post', value=choice_padding)
                tt_id1 = [
                    (len(commonsensetask) + 1) * rstokn + (len(value[1]) + 1) * questons + (len(value[2]) + 1) * rstokn +
                    (len(value[3]) + 1) * questons + (len(value[4]) + 2) * rstokn
                ]
                tt_id1 = sequence.pad_sequences(tt_id1, maxlen=max_length, padding='post', value=pad)

                for i in range(3):

                    mc_token_ids.append(len(ml))
                    input_ids.append(input1[0])
                    lm_labels.append(fakechoice[0])
                    token_type_ids.append(tt_id1[0])

                    input2 = [bos + commonsensetask + rstokn + value[1] + rstokn + value[7 + i] + eos]
                    mc_token_ids.append(len(input2[0]))
                    input2 = sequence.pad_sequences(input2, maxlen=max_length, padding='post', value=pad)
                    input_ids.append(input2[0])

                    choice = [[-1] + [-1] * len(commonsensetask) + [-1] + [-1] * len(value[1]) + [-1] + value[7 + i] + eos]
                    choice = sequence.pad_sequences(choice, maxlen=max_length, padding='post', value=choice_padding)
                    lm_labels.append(choice[0])

                    tt_id2 = [(len(commonsensetask) + 1) * rstokn + (len(value[1]) + 1) * questons + (len(value[7 + i]) + 2) * rstokn]
                    tt_id2 = sequence.pad_sequences(tt_id2, maxlen=max_length, padding='post', value=pad)
                    token_type_ids.append(tt_id2[0])

                    mc_labels.append(value[5])
                    mc_labels.append(value[5])
                    # mc_labels.append(0)


    return input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels

def converting_tokens(data, tokenizer):

    print("Converting tokens to ids ...", flush=True)



    input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = padding_falsesent_choices(data, tokenizer)

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.view((-1, 2) + input_ids.shape[1:])

    mc_token_ids = torch.tensor(mc_token_ids)
    mc_token_ids = mc_token_ids.view((-1, 2) + mc_token_ids.shape[1:])

    lm_labels = torch.tensor(lm_labels)
    lm_labels = lm_labels.view((-1, 2) + lm_labels.shape[1:])

    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.view((-1, 2) + token_type_ids.shape[1:])

    mc_labels = torch.tensor(mc_labels)
    mc_labels = mc_labels.view((-1, 2) + mc_labels.shape[1:])

    pickle.dump(input_ids, open("data/pickle/input_ids.p", "wb"))
    pickle.dump(mc_token_ids, open("data/pickle/mc_token_ids.p", "wb"))
    pickle.dump(lm_labels, open("data/pickle/lm_labels.p", "wb"))
    pickle.dump(token_type_ids, open("data/pickle/token_type_ids.p", "wb"))
    pickle.dump(mc_labels, open("data/pickle/mc_labels.p", "wb"))


    return input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels


def train(model, optimizer, scheduler, train_data, output_dir, num_train_epochs, tokenizer, lm_coef, mc_coef,gradient_accumulation_steps, mgpu, temp=[], valid_data = []):

    training_loss = {}
    evaluation_loss = {}
    global_steps = 0
    for epochs in range(num_train_epochs):
        model.train()
        print("Training start for epoch {}".format(epochs), flush=True)
        nb_tr_steps, tr_loss = 0, 0
        optimizer.zero_grad()
        lm_sub_batch_loss, mc_sub_batch_loss, sub_batch_loss = 0, 0, 0
        print("sub_batch_loss \t lm_sub_batch_loss \t mc_sub_batch_loss")
        for step, batch in (enumerate(train_data)):
            model.train()
            batch = tuple(t.to(device).type(torch.cuda.LongTensor) for t in batch)
            input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch

            lm_loss, mc_loss, *_ = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                mc_labels=mc_labels, lm_labels=lm_labels, task=input_ids[0][0][1]
            )
            mc_loss = mc_loss[0]
            del input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels
            loss = (lm_loss * lm_coef) + (mc_loss * mc_coef)
            loss = loss.mean()
            loss /= gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            lm_sub_batch_loss += lm_loss.item()
            mc_sub_batch_loss += mc_loss.item()
            sub_batch_loss += loss.item()

            if (global_steps + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                # global_steps +=1
                optimizer.zero_grad()
                print("{} \t {} \t {}".format(sub_batch_loss, lm_sub_batch_loss/gradient_accumulation_steps, mc_sub_batch_loss/gradient_accumulation_steps))
                writer.add_scalar('Training batch loss', sub_batch_loss, global_steps+1)
                writer.add_scalar('Training lm batch loss', lm_sub_batch_loss/gradient_accumulation_steps, global_steps+1)
                writer.add_scalar('Training mc batch loss', mc_sub_batch_loss/gradient_accumulation_steps, global_steps+1)
                training_loss[(global_steps+1)] = (sub_batch_loss, lm_sub_batch_loss/gradient_accumulation_steps, mc_sub_batch_loss/gradient_accumulation_steps)
                lm_sub_batch_loss, mc_sub_batch_loss, sub_batch_loss = 0, 0, 0

            if (global_steps + 1) % 800 == 0:
                eval_loss, eval_lm_loss, eval_mc_loss = evaluate_gpt2(model, valid_data)
                print("{} \t {} \t {}".format(eval_mc_loss, eval_lm_loss, eval_mc_loss))
                writer.add_scalar('Eval total loss - 100', eval_loss, (global_steps + 1))
                writer.add_scalar('Eval total LM loss - 100', eval_lm_loss, (global_steps + 1))
                writer.add_scalar('Eval total MC loss - 100', eval_mc_loss, (global_steps + 1))
                evaluation_loss[(global_steps + 1)] = (eval_loss, eval_lm_loss, eval_mc_loss)

                if not os.path.exists(output_dir + '/' + str(global_steps + 1)):
                    os.makedirs(output_dir + '/' + str(global_steps + 1))
                torch.save(model, output_dir + '/' + str(global_steps + 1) + '/' + str(global_steps + 1) + '.pt')
                # model.save_state_dict(output_dir + '/' + str(global_steps + 1))
            global_steps += 1
        print("Epoch Completed at Step Size {}".format(global_steps))
        if not os.path.exists(output_dir + '/' + '_epoch_' + str(epochs)):
            os.makedirs(output_dir + '/' + '_epoch_' + str(epochs))
        torch.save(model, output_dir + '/' + '_epoch_' + str(epochs) + '/' + str(epochs) + '.pt')
        # model.save_state_dict(output_dir + '/' + '_epoch_' + str(epochs))

    pickle.dump(training_loss, open("data/pickle/training_loss-melco-update.p", "wb"))
    pickle.dump(evaluation_loss, open("data/pickle/evaluation_loss-melco-update.p", "wb"))

    return model
def evaluate_gpt2(model, valid_data):

    lm_sub_batch_loss, mc_sub_batch_loss = 0, 0
    model.eval()
    print("\n *************************Evaluation************************************ \n")
    for step, batch in (enumerate(valid_data)):
        batch = tuple(t.to(device).type(torch.cuda.LongTensor) for t in batch)
        input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch

        lm_loss, mc_loss, *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels, task=input_ids[0][0][1]
        )
        del input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels
        lm_sub_batch_loss += lm_loss.item()
        mc_sub_batch_loss += mc_loss[0].item()

    return (lm_sub_batch_loss + mc_sub_batch_loss)/len(valid_data), (lm_sub_batch_loss)/len(valid_data), (mc_sub_batch_loss)/len(valid_data)

def main():
    flag = True
    mgpu = True
    output_dir= 'checkpoints-More-melco-new'
    epochs = 3
    gradient_accumulation_steps = 8
    lm_coef, mc_coef = 1, 0

    token_class = GPT2Tokenizer
    model_Class = MTDNN

    gpt_model = model_Class.from_pretrained('omcs/-Final')
    # gpt_model = model_Class.from_pretrained('gpt2-large')
    gpt_tokenizer = token_class.from_pretrained('omcs/-Final', do_lower_case=True)
    # gpt_tokenizer = token_class.from_pretrained('gpt2-large', do_lower_case=True)

    gpt_model, gpt_tokenizer = Utils.add_special_tokens(gpt_model, gpt_tokenizer, ATTR_TO_SPECIAL_TOKEN)
    gpt_model.to(device)
    #gpt_model = torch.nn.DataParallel(gpt_model, output_device=1, device_ids=[0, 1])

    cache_input_ids, cache_mc_token_ids, cache_lm_labels, cache_token_type_ids, cache_mc_labels = \
            "data/pickle/input_ids.p", "data/pickle/mc_token_ids.p", "data/pickle/lm_labels.p", "data/pickle/token_type_ids.p", "data/pickle/mc_labels.p"
    if flag and os.path.exists(cache_input_ids) and os.path.exists(cache_mc_token_ids) and os.path.exists(
                cache_lm_labels) and os.path.exists(cache_token_type_ids) and os.path.exists(cache_mc_labels):
            print("Token ids loaded from previous processed file ... ", flush=True)
            input_ids, mc_token_ids, lm_labels, token_type_ids, mc_labels = pickle.load(open(cache_input_ids, "rb")), pickle.load(open(cache_mc_token_ids, "rb")), \
                                                                            pickle.load(open(cache_lm_labels, "rb")), pickle.load(open(cache_token_type_ids, "rb")), \
                                                                            pickle.load(open(cache_mc_labels, "rb"))
    else:
        data = data_preprocess()
        input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = converting_tokens(data, gpt_tokenizer)

    temp = [input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels]
    train_data, valid_data = Utils.build_dataloader((input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels))
    train_data, valid_data = Utils.generate_batch(train_data, valid_data, 1)

    t_total = len(train_data) / epochs
    learning_rate, adam_epsilon, weight_decay, warmup_steps = 1e-5, 1e-8, 0, 0

    optimizer, scheduler = Utils.optimizer_generater(gpt_model, learning_rate, adam_epsilon, weight_decay, warmup_steps, t_total)

    model = train(gpt_model, optimizer, scheduler, train_data, output_dir, epochs, gpt_tokenizer, lm_coef, mc_coef, gradient_accumulation_steps,
                  mgpu, temp, valid_data)
    print("End of execution", flush=True)
    output_dir = output_dir + '/' + 'final'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gpt_tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    main()