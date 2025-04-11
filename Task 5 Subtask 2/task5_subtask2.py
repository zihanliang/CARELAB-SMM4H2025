# -------------------------------------------------------------------
# Monkey patch 代码：解决环境中的 TrainingArguments、Trainer、Accelerator 相关问题
# -------------------------------------------------------------------
import os
import sys
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 导入 transformers、accelerate、sklearn 所需模块
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from dataclasses import field

# Monkey patch: 修改 TrainingArguments 的 __init__ 和 __post_init__
try:
    _orig_trainargs_init = TrainingArguments.__init__
    def _patched_trainargs_init(self, *args, **kwargs):
        self._custom_evaluation_strategy = kwargs.pop("evaluation_strategy", "epoch")
        self._custom_save_strategy = kwargs.pop("save_strategy", "epoch")
        if "deepspeed_plugin" in kwargs:
            kwargs.pop("deepspeed_plugin")
        return _orig_trainargs_init(self, *args, **kwargs)
    TrainingArguments.__init__ = _patched_trainargs_init

    def force_trainargs_post_init(self):
        self.evaluation_strategy = IntervalStrategy.EPOCH
        self.save_strategy = SaveStrategy.EPOCH
        if not hasattr(self, "distributed_state"):
            self.distributed_state = None
        if not hasattr(self, "deepspeed_plugin"):
            self.deepspeed_plugin = None
        if not hasattr(self, "fsdp_config") or self.fsdp_config is None:
            self.fsdp_config = {"xla": False}
    TrainingArguments.__post_init__ = force_trainargs_post_init

    print("TrainingArguments monkey patch applied successfully.")
except Exception as e:
    print("Error while monkey patching TrainingArguments:", e)

# Monkey patch: Trainer 的 accelerator_config
try:
    _orig_trainer_create_accel = Trainer.create_accelerator_and_postprocess

    class DummyAccelConfig:
        def __init__(self):
            self.split_batches = False
            self.dispatch_batches = None
            self.even_batches = True
            self.use_seedable_sampler = True
            self.gradient_accumulation_kwargs = {}  # 必须存在
            self.non_blocking = False               # 新增属性
        def to_dict(self):
            return {
                "split_batches": self.split_batches,
                "dispatch_batches": self.dispatch_batches,
                "even_batches": self.even_batches,
                "use_seedable_sampler": self.use_seedable_sampler,
                "gradient_accumulation_kwargs": self.gradient_accumulation_kwargs,
                "non_blocking": self.non_blocking,
            }

    def _patched_trainer_create_accel(self):
        if self.args.accelerator_config is None:
            self.args.accelerator_config = DummyAccelConfig()
        return _orig_trainer_create_accel(self)
    Trainer.create_accelerator_and_postprocess = _patched_trainer_create_accel

    print("Trainer.create_accelerator_and_postprocess monkey patch applied successfully.")
except Exception as e:
    print("Error while monkey patching Trainer:", e)

# Monkey patch: Accelerator.__init__
try:
    _orig_accelerator_init = Accelerator.__init__
    def _patched_accelerator_init(self, *args, **kwargs):
        for key in ["dispatch_batches", "even_batches", "use_seedable_sampler"]:
            kwargs.pop(key, None)
        _orig_accelerator_init(self, *args, **kwargs)
        if not hasattr(self.state, "distributed_type"):
            from accelerate.state import DistributedType
            self.state.distributed_type = DistributedType.NO
    Accelerator.__init__ = _patched_accelerator_init
    print("Accelerator monkey patch applied successfully.")
except Exception as e:
    print("Error while monkey patching Accelerator:", e)

# Monkey patch: AcceleratorState._reset_state（采用 classmethod 形式）
try:
    from accelerate.state import AcceleratorState, DistributedType
    @classmethod
    def _patched_reset_state(cls, *args, **kwargs):
        cls.distributed_type = DistributedType.NO
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AcceleratorState._reset_state = _patched_reset_state
    print("AcceleratorState._reset_state patch applied successfully.")
except Exception as e:
    print("Error while patching AcceleratorState._reset_state:", e)

# -------------------------------------------------------------------
# 以下为模型训练及评估完整代码（使用更大的模型及高级统计学方法提升性能）
# -------------------------------------------------------------------
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 此处需要安装 torchcrf： pip install pytorch-crf
from torchcrf import CRF

###################################
# 1. 定义标签、数据加载和实用函数
###################################
labels = [
    "O", 
    "B-TARGET_ORG", "I-TARGET_ORG", 
    "B-PRODUCT", "I-PRODUCT", 
    "B-INFECTION", "I-INFECTION", 
    "B-SAFETY_INCIDENT", "I-SAFETY_INCIDENT", 
    "B-AFFECTED_NUM", "I-AFFECTED_NUM",
    "B-LOCATION", "I-LOCATION"
]
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

column_to_entity = {
    "organization": "TARGET_ORG",
    "product": "PRODUCT",
    "disease": "INFECTION",
    "cause": "SAFETY_INCIDENT",
    "number_of_people_affected": "AFFECTED_NUM",
    "location": "LOCATION"
}

def load_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df

def preprocess_dataframe(df):
    # 根据需要进行额外的数据清洗，目前直接返回
    return df

def find_all_occurrences(text, substring):
    """返回文本中所有子串 substring 的起始位置列表"""
    starts = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1
    return starts

def get_entities_from_example(example):
    """
    根据 CSV 中每一列（非空）的内容及预定义的列到实体映射关系，
    返回形如 {"start": 起始位置, "end": 结束位置, "label": 实体类型} 的实体列表
    """
    entities = []
    text = example["text"]
    for col, ent_type in column_to_entity.items():
        if col in example and pd.notna(example[col]):
            ent_text = str(example[col]).strip()
            if ent_text:
                start_positions = find_all_occurrences(text, ent_text)
                for st in start_positions:
                    entities.append({
                        "start": st,
                        "end": st + len(ent_text),
                        "label": ent_type
                    })
    return entities

###################################
# 2. 分词与标签对齐
###################################
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, return_offsets_mapping=True)
    all_labels = []
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        text = examples["text"][i]
        example = { key: examples[key][i] for key in examples.keys() }
        entities = get_entities_from_example(example)
        char_labels = ["O"] * len(text)
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            ent_label = ent["label"]
            if 0 <= start < len(text):
                char_labels[start] = "B-" + ent_label
                for j in range(start + 1, end):
                    if j < len(text):
                        char_labels[j] = "I-" + ent_label
        label_ids = []
        for offset in offsets:
            if offset is None or offset[0] == offset[1]:
                label_ids.append(-100)
            else:
                label = char_labels[offset[0]]
                label_ids.append(label2id.get(label, label2id["O"]))
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

###################################
# 3. 序列标注辅助函数
###################################
def extract_event(predicted_labels, tokens):
    """
    根据 BIO 规则将 token 序列拼接为实体，返回形如 {实体类型: [实体文本, ...]} 的字典
    """
    event = defaultdict(list)
    current_entity = None
    current_tokens = []
    for label, token in zip(predicted_labels, tokens):
        if label.startswith("B-"):
            if current_entity is not None:
                event[current_entity].append(" ".join(current_tokens))
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_entity is not None:
            current_tokens.append(token)
        else:
            if current_entity is not None:
                event[current_entity].append(" ".join(current_tokens))
            current_entity = None
            current_tokens = []
    if current_entity is not None:
        event[current_entity].append(" ".join(current_tokens))
    return dict(event)

def convert_gold_to_event(example):
    """
    将真实标注转换为 {实体类型: [实体文本, ...]} 格式
    """
    text = example["text"]
    entities = get_entities_from_example(example)
    event = defaultdict(list)
    for ent in entities:
        label = ent["label"]
        start, end = ent["start"], ent["end"]
        ent_text = text[start:end]
        event[label].append(ent_text)
    return dict(event)

def events_exact_match(pred_event, gold_event):
    """
    若预测事件与真实事件在实体类型及实体文本集合上完全一致，则返回 True
    """
    if set(pred_event.keys()) != set(gold_event.keys()):
        return False
    for key in pred_event:
        if set(pred_event[key]) != set(gold_event[key]):
            return False
    return True

###################################
# 4. 自定义高级 TokenClassification 模型
#    支持 CRF 层以及多种损失（"crf", "focal", "ce"）
###################################
from transformers.modeling_outputs import TokenClassifierOutput

class AdvancedTokenClassificationModel(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        # 默认参数，可通过 from_pretrained 时覆盖
        self.class_weights = torch.ones(config.num_labels)
        self.loss_type = "crf"  # 默认采用 CRF; 可改为 "focal" 或 "ce"
        self.gamma = 2.0
        self.label_smoothing = 0.0
        # 若使用 CRF，则构建 CRF 层
        if self.loss_type == "crf":
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        else:
            self.crf = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 弹出自定义参数，防止传递给父类 __init__
        class_weights = kwargs.pop("class_weights", None)
        loss_type = kwargs.pop("loss_type", "crf")
        gamma = kwargs.pop("gamma", 2.0)
        label_smoothing = kwargs.pop("label_smoothing", 0.0)
        model = super(AdvancedTokenClassificationModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        # 设置自定义参数
        model.class_weights = class_weights if class_weights is not None else torch.ones(model.config.num_labels)
        model.loss_type = loss_type
        model.gamma = gamma
        model.label_smoothing = label_smoothing
        if loss_type == "crf":
            model.crf = CRF(num_tags=model.config.num_labels, batch_first=True)
        else:
            model.crf = None
        return model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 调用父类 forward 得到 logits
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)
        if labels is not None:
            if self.loss_type == "crf":
                # 构造 mask：labels != -100
                mask = labels.ne(-100)
                # 为避免 CRF 出错，将 ignore_index 部分临时设为0
                tags = labels.clone()
                tags[~mask] = 0
                crf_loss = - self.crf(logits, tags, mask=mask, reduction='mean')
                return TokenClassifierOutput(loss=crf_loss, logits=logits)
            else:
                loss = self.compute_loss(TokenClassifierOutput(logits=logits), labels)
                return TokenClassifierOutput(loss=loss, logits=logits)
        else:
            if self.loss_type == "crf":
                # 若未传入 labels，则利用 CRF 进行解码
                mask = attention_mask.bool()
                decoded_tags = self.crf.decode(logits, mask=mask)
                # 将解码结果放入 hidden_states 字段，便于后续评价时使用
                return TokenClassifierOutput(logits=logits, hidden_states=decoded_tags)
            else:
                return TokenClassifierOutput(logits=logits)

    def compute_loss(self, model_outputs, labels):
        logits = model_outputs.logits  # (batch_size, seq_len, num_labels)
        logits = logits.view(-1, self.num_labels)
        labels = labels.view(-1)
        valid_mask = labels.ne(-100)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        logits = logits[valid_mask]
        labels = labels[valid_mask]
        if self.loss_type == "focal":
            ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=self.class_weights.to(logits.device))
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()
        elif self.loss_type == "ce":
            if self.label_smoothing > 0:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device),
                                                       ignore_index=-100,
                                                       label_smoothing=self.label_smoothing)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device),
                                                       ignore_index=-100)
            return loss_fct(logits, labels)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device),
                                                   ignore_index=-100)
            return loss_fct(logits, labels)

###################################
# 5. 自定义 Adversarial Trainer（基于 FGM 对抗训练）
###################################
class AdversarialTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        # 正常前向计算损失
        loss = self.compute_loss(model, inputs)
        loss.backward()
        # FGM 对抗训练
        epsilon = 0.5
        # 假设使用的是 RoBERTa 模型，获取其 word embeddings
        embed_layer = model.roberta.embeddings.word_embeddings
        # 备份原参数梯度方向
        delta = epsilon * torch.sign(embed_layer.weight.grad)
        embed_layer.weight.data.add_(delta)
        # 对抗前向，计算对抗性损失
        adv_loss = self.compute_loss(model, inputs)
        adv_loss.backward()
        # 恢复 embeddings 参数
        embed_layer.weight.data.sub_(delta)
        # 返回总损失
        total_loss = loss + adv_loss
        return total_loss.detach()

###################################
# 6. 辅助函数：计算类别权重（基于 token 层统计）
###################################
def compute_class_weights(dataset, num_labels):
    counter = Counter()
    for example in dataset:
        for label in example["labels"]:
            if label != -100:
                counter[label] += 1
    weights = []
    for i in range(num_labels):
        count = counter[i]
        weights.append(1.0 / count if count > 0 else 1.0)
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum()  # 归一化
    return weights

###################################
# 7. 修改 compute_metrics：支持 CRF 解码后的输出
###################################
def compute_metrics(p):
    predictions, labels_true = p
    # 若预测结果为列表（即 CRF 解码结果），则直接使用；否则进行 argmax
    if isinstance(predictions, list):
        true_predictions = predictions
    else:
        predictions = np.argmax(predictions, axis=2)
        true_predictions = []
        for pred, label in zip(predictions, labels_true):
            pred_labels = []
            label_list = []
            for p_idx, l_idx in zip(pred, label):
                if l_idx != -100:
                    pred_labels.append(id2label[p_idx])
                    label_list.append(id2label[l_idx])
            true_predictions.append(pred_labels)
    true_labels = []
    for label in labels_true:
        label_list = []
        for l in label:
            if l != -100:
                label_list.append(id2label[l])
        true_labels.append(label_list)
    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
         "precision": results["overall_precision"],
         "recall": results["overall_recall"],
         "f1": results["overall_f1"],
         "accuracy": results["overall_accuracy"],
    }

###################################
# 8. 主流程：训练、评估与高级分析（混淆矩阵、分类报告等）
###################################
if __name__ == "__main__":
    # 数据文件路径
    train_path = "SMM4H-2025-Task5-Train_subtask2.csv"
    val_path = "SMM4H-2025-Task5-Validation_subtask2.csv"
    
    # 加载并预处理数据
    train_df, val_df = load_data(train_path, val_path)
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    
    # 转换为 Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    ###################################
    # 9. 初始化 Tokenizer 与模型（使用 roberta-large）
    ###################################
    model_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 分词和标签对齐（remove_columns 转为 list 以避免错误）
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=list(train_df.columns))
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=list(val_df.columns))
    
    # 计算类别权重（基于训练集 token 层标签分布）
    class_weights = compute_class_weights(train_dataset, num_labels=len(labels))
    
    # 加载配置，并实例化自定义模型
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    model = AdvancedTokenClassificationModel.from_pretrained(
        model_name,
        config=config,
        class_weights=class_weights,
        loss_type="crf",      # 使用 CRF 层
        gamma=2.0,
        label_smoothing=0.0
    )
    
    ###################################
    # 10. 定义训练参数与自定义 AdversarialTrainer（未启用早停功能）
    ###################################
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    ###################################
    # 11. 开始训练与评估（Token-Level 及 Event-Level）
    ###################################
    trainer.train()
    
    # Token-Level 评估
    metrics = trainer.evaluate()
    print("Token-Level Evaluation Metrics:")
    print(metrics)
    
    # Event-Level Exact Match 评估
    exact_match_count = 0
    total_examples = len(val_df)
    device = model.device
    for i in tqdm(range(total_examples), desc="Evaluating event extraction"):
        example = val_df.iloc[i]
        text = example["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: t.to(device) for k, t in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # 若使用 CRF层，预测结果存于 hidden_states 字段
        if model.loss_type == "crf":
            pred_ids = outputs.hidden_states[0]  # decoded list-of-lists
        else:
            pred_ids = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy().tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        filtered_tokens = []
        filtered_preds = []
        for token, p in zip(tokens, pred_ids):
            if token in tokenizer.all_special_tokens:
                continue
            # 若 p 为整数则转换
            if isinstance(p, int):
                filtered_preds.append(id2label[p])
            else:
                filtered_preds.append(id2label[p])
            filtered_tokens.append(token)
        pred_event = extract_event(filtered_preds, filtered_tokens)
        gold_event = convert_gold_to_event(example)
        if events_exact_match(pred_event, gold_event):
            exact_match_count += 1
    event_exact_match_rate = exact_match_count / total_examples
    print(f"Event-Level Exact Match Rate: {event_exact_match_rate:.4f}")
    
    ###################################
    # 12. 高级分析：混淆矩阵与分类报告（过滤掉 "O" 标签）
    ###################################
    flat_preds = []
    flat_labels = []
    for batch in trainer.get_eval_dataloader():
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        # 根据是否使用 CRF选择预测方式
        if model.loss_type == "crf":
            batch_preds = outputs.hidden_states[0]
        else:
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels_batch = batch["labels"].cpu().numpy()
        # 若为 CRF，则 batch_preds 为 list-of-lists
        if isinstance(batch_preds, list):
            for pred_seq, l_seq in zip(batch_preds, labels_batch):
                for p, l in zip(pred_seq, l_seq):
                    if l != -100:
                        flat_preds.append(p)
                        flat_labels.append(l)
        else:
            for p_seq, l_seq in zip(batch_preds, labels_batch):
                for p, l in zip(p_seq, l_seq):
                    if l != -100:
                        flat_preds.append(p)
                        flat_labels.append(l)
    filtered_preds, filtered_labels = [], []
    o_id = label2id["O"]
    for p, l in zip(flat_preds, flat_labels):
        if p != o_id and l != o_id:
            filtered_preds.append(p)
            filtered_labels.append(l)
    ALL_LABELS = [
        "O", 
        "B-TARGET_ORG", "I-TARGET_ORG", 
        "B-PRODUCT", "I-PRODUCT", 
        "B-INFECTION", "I-INFECTION", 
        "B-SAFETY_INCIDENT", "I-SAFETY_INCIDENT", 
        "B-AFFECTED_NUM", "I-AFFECTED_NUM",
        "B-LOCATION", "I-LOCATION"
    ]
    entity_only = [lab for lab in ALL_LABELS if lab != "O"]
    entity_ids = [label2id[lab] for lab in entity_only]
    
    cm_entities = confusion_matrix(filtered_labels, filtered_preds, labels=entity_ids)
    print("\nConfusion Matrix (Gold, Pred ≠ O):")
    print(cm_entities)
    
    print("\nClassification Report (Gold, Pred ≠ O):")
    print(classification_report(
        filtered_labels,
        filtered_preds,
        labels=entity_ids,
        target_names=entity_only,
        zero_division=0
    ))
    
    ###################################
    # 13. 实体级（Chunk-Level） F1 评估（基于 seqeval）
    ###################################
    preds = []
    labels_true = []
    for batch in trainer.get_eval_dataloader():
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        if model.loss_type == "crf":
            batch_preds = outputs.hidden_states[0]
        else:
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels_batch = batch["labels"].cpu().numpy()
        preds.extend(batch_preds)
        labels_true.extend(labels_batch)
    
    label_seqeval_preds = []
    label_seqeval_trues = []
    # 判断 preds 是否为 list（即 CRF 输出）或 numpy 数组
    if isinstance(preds, list):
        for pred_seq, label_seq in zip(preds, labels_true):
            pred_labels_str = []
            gold_labels_str = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                pred_labels_str.append(id2label[p])
                gold_labels_str.append(id2label[l])
            label_seqeval_preds.append(pred_labels_str)
            label_seqeval_trues.append(gold_labels_str)
    else:
        for pred_seq, label_seq in zip(preds, labels_true):
            pred_labels_str = []
            gold_labels_str = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                pred_labels_str.append(id2label[p])
                gold_labels_str.append(id2label[l])
            label_seqeval_preds.append(pred_labels_str)
            label_seqeval_trues.append(gold_labels_str)
    seqeval_metric = evaluate.load("seqeval")
    chunk_results = seqeval_metric.compute(
        predictions=label_seqeval_preds,
        references=label_seqeval_trues
    )
    print("\nChunk-Level (Entity-Level) Seqeval Results:")
    print(chunk_results)
