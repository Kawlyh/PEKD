from typing import Optional, List
import torch
from peft import PeftModelForSequenceClassification
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertForSequenceClassification
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler


def distill_bert(teacher_model):
    """
    Distillates a bert (teacher_model)
    The student model has the same configuration, except for the number of hidden layers, which is // 2.
    The student layers are initialized by copying one out of two layers of the teacher, starting with layer 0.
    The head of the teacher is also copied.
    """

    # Get teacher configuration as a dictionary
    configuration = teacher_model.config.to_dict()
    # Half the number of hidden layer
    configuration['num_hidden_layers'] //= 4
    # Convert the dictionary to the student configuration
    configuration = BertConfig.from_dict(configuration)
    # Create uninitialized student model
    # print(type(teacher_model))
    # print(teacher_model.peft_config)
    # student_model = type(teacher_model)(configuration)
    student_model = BertForSequenceClassification(configuration)
    # Initialize the student's weights
    distill_bert_weights(teacher=teacher_model, student=student_model)
    # Return the student model
    return student_model

def distill_bert_weights(teacher, student):
    """
    Recursively copied the weights of the (teacher) to the (student).
    This function is meant to be first called on a BertFor... model, but is then called on every children of the model recursively.
    The only part that's not fully copied is the encoder, of which only half is copied.
    """

    # If the part is an entire RoBERTa model or a RobertaFor..., unpack and interate
    if isinstance(teacher, BertModel) or type(teacher).__name__.startswith('BertFor'):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_bert_weights(teacher_part, student_part)
    # Else if the part is an encoder, copy one out of every layer
    elif isinstance(teacher, BertEncoder):
        teacher_encoding_layers = [layer for layer in next(teacher.children())]
        student_encoding_layers = [layer for layer in next(student.children())]
        for i in range(len(student_encoding_layers)):
            student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
    # Else the part is a head or something else, copy the state_dict
    else:
        student.load_state_dict(teacher.state_dict(),strict=False)
        # student.load_state_dict(teacher.state_dict())