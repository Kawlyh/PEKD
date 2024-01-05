import os
import warnings
from datasets import load_dataset, load_from_disk
import logging
from transformers import DataCollatorWithPadding, BertForSequenceClassification, DataProcessor, InputExample
from transformers import glue_processors as processors
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import glue_output_modes as output_modes, AutoTokenizer
import numpy as np
import torch
from transformers.data.metrics import DEPRECATION_WARNING

logger = logging.getLogger(__name__)

glue_task="mnli_mismatched"
print("task is :"+ glue_task)
#'cola', 'sst-b','sst-2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched',
# 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device is :"+ device)

if glue_task not in ["ax","mnli_matched","mnli_mismatched"]:
    model_name_or_path="./checkpoint/best-student-model-"+glue_task+".pt"
else:
    # model_name_or_path="./checkpoint/best-student-model-mnli-matched.pt"
    model_name_or_path="/home/wangyukun/workspace/kd/tinybert_model/layer6"
print("run model:"+model_name_or_path)
checkpoint = model_name_or_path

output_dir="./result/"
print("output dir:"+output_dir)

if glue_task in["sst-2","ax","mnli_matched","mnli_mismatched"]:
    output_mode = "classification"
else:
    output_mode = output_modes[glue_task]
print("now the task is :"+output_mode)

batchnum=32
print("eval batch size is:"+str(batchnum))

maxlength=128

raw_datasets = load_from_disk(f"/home/wangyukun/workspace/kd/glue/{glue_task}")


class AxProcessor(DataProcessor):
    """Processor for the diagnostic data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "diagnostic.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]  # sentence1
            text_b = line[2]  # sentence2
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def makedata():
    my_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def preprocess_function(examples):
        if glue_task in ["sst-2","cola"]:
            return my_tokenizer(examples['sentence'], truncation=True, padding="max_length",
                                max_length=maxlength)
        elif glue_task in ["qnli"]:
            return my_tokenizer(examples['sentence'], examples['question'], truncation=True, padding="max_length",
                                max_length=maxlength)
        elif glue_task == "qqp":
            return my_tokenizer(examples['question1'], examples['question2'], truncation=True, padding="max_length",
                                max_length=maxlength)
        elif glue_task in ["mnli", "mnli_mismatched", "mnli_matched","ax"]:
            return my_tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding="max_length",
                                max_length=maxlength)
        else:
            return my_tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length",
                                max_length=maxlength)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=my_tokenizer)
    tokenized_datasets=tokenized_datasets['test']
    if glue_task in ["sst-2","cola"]:
        tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'idx','label'])
    elif glue_task in ["qnli"]:
        tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'question', 'idx','label'])
    elif glue_task=="qqp":
        tokenized_datasets = tokenized_datasets.remove_columns(['question1','question2',  'idx','label'])
    elif glue_task in ["mnli", "mnli_mismatched", "mnli_matched","ax"]:
        tokenized_datasets = tokenized_datasets.remove_columns(['premise', 'hypothesis',  'idx','label'])
    else:
        tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx','label'])

    test_dataloader = DataLoader(tokenized_datasets, shuffle=False, batch_size=batchnum,
                                  collate_fn=data_collator)


    return test_dataloader


def test(model):
    eval_outputs_dirs = output_dir
    test_dataloader = makedata()
    model.eval()
    preds=None
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            # print(batch)
            outputs = model(**batch)
            logits = outputs.logits
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    if glue_task not in["ax","mnli_matched","mnli_mismatched"]:
        processor = processors[glue_task]()
    elif glue_task == "mnli_matched":
        processor = processors["mnli"]()
    elif glue_task == "mnli_mismatched":
        processor = processors["mnli-mm"]()
    else:
        processors['ax'] = AxProcessor
        processor = processors[glue_task]()
    label_list = processor.get_labels()
    label_map = {i: label for i, label in enumerate(label_list)}
    output_eval_file = os.path.join(eval_outputs_dirs, glue_task.upper() + ".tsv")
    with open(output_eval_file, "w") as writer:
        print("***** Predict results *****")
        writer.write("index\tprediction\n")
        for index, pred in enumerate(tqdm(preds)):
            if glue_task == 'sts-b':
                pred = round(pred, 3)
                if pred > 5.:
                    pred = 5.000
            else:
                pred = label_map[pred]
            writer.write("%s\t%s\n" % (index, str(pred)))
    print("final result saved!")
    print("finished!")


if __name__ == "__main__":
    if glue_task not in ["ax","mnli_matched","mnli_mismatched"]:
        model =torch.load(checkpoint)
    else:
        model=BertForSequenceClassification.from_pretrained("/home/wangyukun/workspace/kd/tinybert_model/layer6",num_labels=3)
    model.to(device)
    test(model)
