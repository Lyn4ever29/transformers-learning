from seqeval.metrics import classification_report
import torch
from tqdm import tqdm
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

class Trainer:
    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 save_step=500,
                 dev_loader=None,
                 test_loader=None,
                 epochs=1,
                 device="cpu",
                 id2label=None,
                 t_total_num=0,
                 optimizer_args=None):
        self.schedule = None
        self.optimizer = None
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.id2label = id2label
        self.save_step = save_step
        self.total_step = len(self.train_loader) * self.epochs
        self.t_total_num = t_total_num
        self.optimizer_args = optimizer_args

        self.build_optimizer_and_scheduler()

    def train(self):
        global_step = 1
        epoch_loss = 0
        for epoch in range(1, self.epochs + 1):
            for step, batch_data in enumerate(tqdm(self.train_loader)):
                self.model.train()
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]
                logits, _labels, loss = self.model(input_ids, attention_mask, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
                global_step += 1
                epoch_loss = loss
                if global_step % self.save_step == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_ner.bin"))
            print(f"【train】{epoch}/{self.epochs} {global_step}/{self.total_step} loss:{epoch_loss.item()}")

        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_ner.bin"))

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "pytorch_model_ner.bin")))
        self.model.eval()
        preds = []
        trues = []
        for step, batch_data in enumerate(tqdm(self.test_loader)):
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]
            logits, _labels, loss = self.model(input_ids, attention_mask, labels)
            attention_mask = attention_mask.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = sum(attention_mask[i])
                logit = logits[i][1:length]
                logit = [self.id2label[i] for i in logit]
                label = labels[i][1:length]
                label = [self.id2label[i] for i in label]
                preds.append(logit)
                trues.append(label)

        report = classification_report(trues, preds)
        return report

    # 优化函数
    def build_optimizer_and_scheduler(self):
        args=self.optimizer_args
        model = self.model
        t_total_num = self.t_total_num
        module = (
            model.module if hasattr(model, "module") else model
        )

        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            # print(name)
            if space[0] == 'bert_module' or space[0] == "bert":
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': args.bert_learning_rate},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': args.crf_learning_rate},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.warmup_proportion * t_total_num), num_training_steps=t_total_num
        )

        self.optimizer=optimizer
        self.schedule=scheduler
