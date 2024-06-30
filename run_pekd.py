import numpy as np
from datasets import load_from_disk
import copy
import gc
from datasets import load_dataset, load_from_disk
from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function

from makestudent import distill_bert
import evaluate
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, AdamW, \
    BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
from thop import profile

glue_task = "rte"
print("\n task is :" + glue_task)
# 'cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax'

batch_num = 16
lr_t = 3e-4  #
lr_s = 1e-4  # real stu
lr_fast = 1e-4  # fast stu
epochs = 20
Temp = 5
alpha = 0.05  # softloss for all
beta_1 = 0.05  # mseloss fast stu
beta_2 = 0.05  # mseloss real stu
sita = 0.05  # retrain
checkpoint = ""  # teacher checkpoint
ck_stu = ""
ck_metric = "./metrics/glue"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device is :" + device)

from_pretrained = True
print("used tinybert:" + str(from_pretrained))

use_meta = True
print("used meta:" + str(use_meta))

use_lora = True
print("used lora:" + str(use_lora))

use_adalora = False
print("used adalora:" + str(use_adalora))

use_skip = True
print("used skip:" + str(use_skip))

if glue_task != "mnli":
    # Path = f"./workspace/pekd/checkpoint/best-student-model-{glue_task}.pt"
    Path = f"./checkpoint/best-student-model-{glue_task}.pt"
elif glue_task == "mnli":
    Path1 = f"./checkpoint/best-student-model-mnli-matched.pt"
    Path2 = f"./checkpoint/best-student-model-mnli-mismatched.pt"

raw_datasets = load_from_disk(f"./glue

/{glue_task}")

if glue_task == "sst2":
    label_num = 2
    maxlength = 66
if glue_task == "qqp":
    # question1', 'question2', 'label', 'idx
    label_num = 2
    maxlength = 118
if glue_task == "rte":
    label_num = 2
    maxlength = 250
if glue_task == "mrpc":
    label_num = 2
    maxlength = 103
if glue_task == "qnli":
    label_num = 2
    maxlength = 512
if glue_task == "mnli":
    label_num = 3
    maxlength = 140
if glue_task == "stsb":
    # features: ['sentence1', 'sentence2', 'label', 'idx'],
    maxlength = 128
    label_num = 1
if glue_task == "wnli":
    maxlength = 128
    label_num = 2
if glue_task == "cola":
    maxlength = 128
    label_num = 2


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def KD_loss(logit_pred_student, teacher_pred, t):
    ps = F.log_softmax(logit_pred_student / t, dim=1)
    pt = F.softmax(teacher_pred / t, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(ps, pt) * (t ** 2)


def makedata():
    my_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def preprocess_function(examples):
        if glue_task in ["sst2", "cola"]:
            return my_tokenizer(examples['sentence'], truncation=True, padding="max_length",
                                max_length=maxlength)
        elif glue_task in ["qnli"]:
            return my_tokenizer(examples['sentence'], examples['question'], truncation=True, padding="max_length",
                                max_length=maxlength)
        elif glue_task == "qqp":
            return my_tokenizer(examples['question1'], examples['question2'], truncation=True, padding="max_length",
                                max_length=maxlength)
        elif glue_task == "mnli":
            return my_tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding="max_length",
                                max_length=maxlength)
        else:
            return my_tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length",
                                max_length=maxlength)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=my_tokenizer)

    if glue_task in ["sst2", "cola"]:
        tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'idx'])
    elif glue_task in ["qnli"]:
        tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'question', 'idx'])
    elif glue_task == "qqp":
        tokenized_datasets = tokenized_datasets.remove_columns(['question1', 'question2', 'idx'])
    elif glue_task == "mnli":
        tokenized_datasets = tokenized_datasets.remove_columns(['premise', 'hypothesis', 'idx'])
    else:
        tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=batch_num,
                                  collate_fn=data_collator)
    if glue_task != "mnli":
        val_dataloader = DataLoader(tokenized_datasets['validation'], shuffle=False, batch_size=batch_num,
                                    collate_fn=data_collator)
        return train_dataloader, val_dataloader
    elif glue_task == "mnli":
        val_dataloader_m = DataLoader(tokenized_datasets['validation_matched'], shuffle=False, batch_size=batch_num,
                                      collate_fn=data_collator)
        val_dataloader_mm = DataLoader(tokenized_datasets['validation_mismatched'], shuffle=False, batch_size=batch_num,
                                       collate_fn=data_collator)
        return train_dataloader, val_dataloader_m, val_dataloader_mm


def addlora(teacher_model, student_model):
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS

    )
    print("add LoRA adaptor for teacher model")
    teacher_model = get_peft_model(teacher_model, lora_config)
    print(f"******teacher model parameters after LoRA:", get_parameter_number(teacher_model), "******")

    print("add LoRA adaptor for student model")
    student_model = get_peft_model(student_model, lora_config)
    print(f"******student model parameters after LoRA:", get_parameter_number(student_model), "******")

    return teacher_model, student_model


from peft import get_peft_model, TaskType, AdaLoraConfig


def get_ada_lora(teacher_model, student_model):
    adalora_config = AdaLoraConfig(
        peft_type="ADALORA",
        task_type="TaskType.SEQ_CLS",
        target_r=64,
        init_r=32,
        lora_alpha=32,
        target_modules=["value", 'key', 'query'],
        lora_dropout=0.01,
    )
    print("add adaLoRA adaptor for teacher model")
    teacher_model = get_peft_model(teacher_model, adalora_config)
    print(f"******teacher model parameters after adaLoRA:", get_parameter_number(teacher_model), "******")

    print("add adaLoRA adaptor for student model")
    student_model = get_peft_model(student_model, adalora_config)
    print(f"******student model parameters after adaLoRA:", get_parameter_number(student_model), "******")

    return teacher_model, student_model


def mypkd(teacher_outputs, student_outputs):
    mse_func = nn.MSELoss()
    teacher_all_hidden = teacher_outputs.hidden_states
    student_all_hidden = student_outputs.hidden_states
    if use_skip:
        teacher_layers = teacher_all_hidden[1].detach()  # start at 1 layer
        student_layers = student_all_hidden[0].detach()  # start at 0 layer
        teacher_layers_len = len(teacher_all_hidden)  # 13
        student_layers_len = len(student_all_hidden)  # 7
        ops = int((teacher_layers_len - 1) / (student_layers_len - 1))
        w = 2
        count = 1
        for j in range(2, teacher_layers_len):  # don't need last pool layer
            w += ops
            if w < teacher_layers_len and count <= student_layers_len:
                teacher_layers = torch.cat((teacher_layers, teacher_all_hidden[w].detach()), 0)
                count += 1
        for k in range(1, student_layers_len - 1):
            student_layers = torch.cat((student_layers, student_all_hidden[k]), 0)
    else:
        student_layers = student_all_hidden[0].detach()  # start at 0 layer
        student_layers_len = len(student_all_hidden)  # 7
        for k in range(1, student_layers_len - 1):
            student_layers = torch.cat((student_layers, student_all_hidden[k]), 0)

        teacher_layers = teacher_all_hidden[6].detach()  # start at 6 layer
        teacher_layers_len = len(teacher_all_hidden)  # 13
        for j in range(7, teacher_layers_len - 1):
            teacher_layers = torch.cat((teacher_layers, teacher_all_hidden[j].detach()), 0)

    mse_loss = mse_func(teacher_layers, student_layers)

    return mse_loss


def doMetatrain(teacher_model, student_model, train_dataloader, val_dataloader=None, val_dataloader_m=None,
                val_dataloader_mm=None):
    print("ready go!")
    HardLoss = nn.CrossEntropyLoss()
    temp1 = 0.0
    temp2 = 0.0
    temp3 = 0.0
    temp4 = 0.0
    for i in range(epochs):
        teacher_epoch_loss = 0.0
        num = 0
        student_epoch_loss = 0.0
        # train
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # online fine-tun teacher
            teacher_model.train()
            teacher_model.cuda()
            teacher_outputs = teacher_model(**batch)
            loss1 = teacher_outputs.loss
            loss1.backward()
            optimizer_teacher.step()
            optimizer_teacher.zero_grad()
            teacher_epoch_loss = loss1 * 0.5 + teacher_epoch_loss

            # kd train fast student
            fast_model = copy.deepcopy(student_model)
            optimizer_fast = AdamW(student_model.parameters(), lr=lr_fast, no_deprecation_warning=True)
            # fast_scheduler = get_linear_schedule_with_warmup(optimizer_fast, num_warmup_steps=warmup_steps,
            #                                                     num_training_steps=t_total)
            fast_model.train()
            fast_model.cuda()
            teacher_logits = teacher_outputs[1].detach()
            fast_outputs = fast_model(**batch)
            fast_logits = fast_outputs[1]
            fast_soft_loss = KD_loss(fast_logits, teacher_logits, Temp)
            if glue_task not in ["stsb"]:
                fast_hard_loss = HardLoss(fast_logits, batch["labels"])
            elif glue_task in ["stsb"]:
                mse_func = nn.MSELoss()
                fast_hard_loss = mse_func(fast_logits.squeeze(), batch["labels"].squeeze())
            fast_mse_loss = mypkd(teacher_outputs, fast_outputs)
            loss2 = (1 - alpha) * fast_hard_loss + alpha * fast_soft_loss + beta_1 * fast_mse_loss
            loss2.backward()
            optimizer_fast.step()
            optimizer_fast.zero_grad()

            # eval fast studnet
            fast_model.eval()
            with torch.no_grad():
                new_fast_outputs = fast_model(**batch)

            # retrain teacher
            retrain_teacher_outputs = teacher_model(**batch)
            retrain_mse_loss = mypkd(retrain_teacher_outputs, new_fast_outputs)
            loss_retrain = retrain_teacher_outputs.loss + sita * retrain_mse_loss
            loss_retrain.backward()
            optimizer_teacher.step()
            # teacher_scheduler.step()
            optimizer_teacher.zero_grad()
            teacher_epoch_loss = loss_retrain * 0.5 + teacher_epoch_loss
            ## delete fast
            # del fast_model, optimizer_fast,fast_scheduler
            del fast_model, optimizer_fast
            torch.cuda.empty_cache()
            gc.collect()

            # online eval teacher
            teacher_model.eval()
            with torch.no_grad():
                new_teacher_outputs = teacher_model(**batch)

            # kd train real student
            student_model.train()
            student_model.cuda()
            new_teacher_logits = new_teacher_outputs[1].detach()
            student_outputs = student_model(**batch)
            student_logits = student_outputs[1]
            soft_loss = KD_loss(student_logits, new_teacher_logits, Temp)
            if glue_task not in ["stsb"]:
                hard_loss = HardLoss(student_logits, batch["labels"])
            elif glue_task in ["stsb"]:
                mse_func = nn.MSELoss()
                hard_loss = mse_func(student_logits.squeeze(), batch["labels"].squeeze())
            mse_loss = mypkd(new_teacher_outputs, student_outputs)
            loss3 = (1 - alpha) * hard_loss + alpha * soft_loss + beta_2 * mse_loss
            loss3.backward()
            optimizer_student.step()
            optimizer_student.zero_grad()
            student_epoch_loss = loss3 + student_epoch_loss

            num = 1 + num

        print(f"epoch-{i}-teacher_average_loss:", teacher_epoch_loss / num)
        print(f"epoch-{i}-kd-train-student_average_loss:", student_epoch_loss / num)
        print(f"epoch-{i}-teacher's-retrain-mseloss:{retrain_mse_loss}")
        print(f"epoch-{i}-student's-softloss:{soft_loss}")
        print(f"epoch-{i}-student's-hardloss:{hard_loss}")
        print(f"epoch-{i}-student's-mseloss:{mse_loss}")

        # eval
        # teacher_model.eval()
        student_model.eval()
        if glue_task != "mnli":
            for batch in tqdm(val_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs1 = teacher_model(**batch)
                    outputs2 = student_model(**batch)
                    logits1 = outputs1.logits
                    logits2 = outputs2.logits
                    if glue_task not in ["stsb"]:
                        predictions1 = torch.argmax(logits1, dim=1)
                        predictions2 = torch.argmax(logits2, dim=1)
                    elif glue_task in ["stsb"]:
                        predictions1 = logits1.squeeze()
                        predictions2 = logits2.squeeze()
                    teacher_metric.add_batch(predictions=predictions1, references=batch["labels"])
                    student_metric.add_batch(predictions=predictions2, references=batch["labels"])

            score1 = teacher_metric.compute()
            score2 = student_metric.compute()
            print(f"epoch-{i}-teacher-score:{score1}")
            print(f"epoch-{i}-student-score:{score2}")

            if glue_task in [
                "sst2",
                "qnli",
                "rte",
                "wnli", ]:
                if score1['accuracy'] > temp1:
                    temp1 = score1['accuracy']
                if score2['accuracy'] > temp2:
                    torch.save(student_model, Path)
                    temp2 = score2['accuracy']
            elif glue_task in ["qqp",
                               "mrpc"]:
                if score1['accuracy'] > temp1:
                    temp1 = score1['accuracy']
                    temp3 = score1['f1']
                if score2['accuracy'] > temp2:
                    torch.save(student_model, Path)
                    temp2 = score2['accuracy']
                    temp4 = score2['f1']
            elif glue_task in ["stsb"]:
                if score1['pearson'] > temp1:
                    temp1 = score1['pearson']
                    temp3 = score1['spearmanr']
                if score2['pearson'] > temp2:
                    torch.save(student_model, Path)
                    temp2 = score2['pearson']
                    temp4 = score2['spearmanr']
            elif glue_task == "cola":
                if score1['matthews_correlation'] > temp1:
                    temp1 = score1['matthews_correlation']
                if score2['matthews_correlation'] > temp2:
                    torch.save(student_model, Path)
                    temp2 = score2['matthews_correlation']

        elif glue_task == "mnli":
            for batch in tqdm(val_dataloader_m):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs1 = teacher_model(**batch)
                    outputs2 = student_model(**batch)
                    logits1 = outputs1[1]
                    logits2 = outputs2[1]
                    predictions1 = torch.argmax(logits1, dim=-1)
                    predictions2 = torch.argmax(logits2, dim=-1)
                    teacher_metric.add_batch(predictions=predictions1, references=batch["labels"])
                    student_metric.add_batch(predictions=predictions2, references=batch["labels"])

            score1_m = teacher_metric.compute()
            score2_m = student_metric.compute()
            print(f"epoch-{i}-teacher-accuracy-matched:{score1_m}")
            print(f"epoch-{i}-student-accuracy-matched:{score2_m}")

            for batch in tqdm(val_dataloader_mm):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs1 = teacher_model(**batch)
                    outputs2 = student_model(**batch)
                    logits1 = outputs1[1]
                    logits2 = outputs2[1]
                    predictions1 = torch.argmax(logits1, dim=-1)
                    predictions2 = torch.argmax(logits2, dim=-1)
                    teacher_metric.add_batch(predictions=predictions1, references=batch["labels"])
                    student_metric.add_batch(predictions=predictions2, references=batch["labels"])

            score1_mm = teacher_metric.compute()
            score2_mm = student_metric.compute()
            print(f"epoch-{i}-teacher-accuracy-mismatched:{score1_mm}")
            print(f"epoch-{i}-student-accuracy-mismatched:{score2_mm}")

            if score1_m['accuracy'] > temp1:
                temp1 = score1_m['accuracy']
            if score2_m['accuracy'] > temp2:
                torch.save(student_model, Path1)
                temp2 = score2_m['accuracy']
            if score1_mm['accuracy'] > temp3:
                temp3 = score1_mm['accuracy']
            if score2_mm['accuracy'] > temp4:
                torch.save(student_model, Path2)
                temp4 = score2_mm['accuracy']

    print("KD finished!")

    if glue_task in [
        "sst2",
        "qnli",
        "rte",
        "wnli",
        "cola"]:
        print(f"teacher's best score is {temp1}")
        print(f"student's best score is {temp2}")
    elif glue_task in ["qqp",
                       "mrpc"]:
        print(f"teacher's best score is acc {temp1} and f1 {temp3}")
        print(f"student's best score is acc {temp2} and f1 {temp4}")
    elif glue_task in ["stsb"]:
        print(f"teacher's best score is pearson {temp1} and Spearman {temp3}")
        print(f"student's best score is pearson {temp2} and Spearman {temp4}")
    elif glue_task in ["mnli"]:
        print(f"teacher's best score for matched is acc {temp1}")
        print(f"student's best score for matched is acc {temp2}")
        print(f"teacher's best score for mismatched is acc {temp3}")
        print(f"student's best score for mismatched is acc {temp4}")


def calculate_lora_flops(input_dim, output_dim, rank):
    # FLOPs for the low-rank approximation
    flops_A = input_dim * rank
    flops_B = rank * output_dim
    total_flops = flops_A + flops_B
    return total_flops


def calculate_bert_lora_flops(input_dim, hidden_dim, num_heads, seq_length, rank, num_layers):
    total_flops = 0

    # Each layer has multiple attention heads
    for _ in range(num_layers):
        for _ in range(num_heads):
            # Calculate FLOPs for Q, K, V (only Q and K are modified by LoRA)
            total_flops += 2 * calculate_lora_flops(input_dim, hidden_dim // num_heads, rank)

    # Scale by sequence length
    total_flops *= seq_length

    return total_flops


if __name__ == "__main__":
    teacher_model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=label_num,
                                                                  output_hidden_states=True)
    if from_pretrained:
        student_model = BertForSequenceClassification.from_pretrained(ck_stu, num_labels=label_num,
                                                                      output_hidden_states=True)
    else:
        student_model = distill_bert(teacher_model)
    print(f"******initial teacher model parameterts:", get_parameter_number(teacher_model), "******")
    print(f"******initial student model parameterts:", get_parameter_number(student_model), "******")
    if use_lora:
        teacher_model, student_model = addlora(teacher_model, student_model)
    else:
        pass
    if use_adalora:
        teacher_model, student_model = get_ada_lora(teacher_model, student_model)
    else:
        pass
    teacher_metric = evaluate.load(ck_metric, glue_task)
    student_metric = evaluate.load(ck_metric, glue_task)
    optimizer_teacher = AdamW(teacher_model.parameters(), lr=lr_t, no_deprecation_warning=True)
    optimizer_student = AdamW(student_model.parameters(), lr=lr_s, no_deprecation_warning=True)
    if glue_task != "mnli":
        train_dataloader, val_dataloader = makedata()
        doMetatrain(teacher_model=teacher_model, student_model=student_model, train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader, val_dataloader_m=None, val_dataloader_mm=None)
    elif glue_task == "mnli":
        train_dataloader, val_dataloader_m, val_dataloader_mm = makedata()
        doMetatrain(teacher_model=teacher_model, student_model=student_model, train_dataloader=train_dataloader,
                    val_dataloader=None, val_dataloader_m=val_dataloader_m, val_dataloader_mm=val_dataloader_mm)
