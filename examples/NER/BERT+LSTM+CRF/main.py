import os
import json
import torch
from Trainer import Trainer
from config import NerConfig
from model import BertNer
from data_loader import NerDataset

from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer,AutoTokenizer

from predict import Predictor

labels_list = ['O', 'B-HCCX', 'I-HCCX', 'B-HPPX', 'I-HPPX', 'B-XH', 'I-XH', 'B-MISC', 'I-MISC']
num_labels = len(labels_list)
label2id = {l: i for i, l in enumerate(labels_list)}
id2labels = {i: l for i, l in enumerate(labels_list)}
# 读取数据
def read_data(file):
    lists = []
    with open(file, 'r') as f:
        lines = f.readlines()
        id = 0
        tokens = []
        ner_tags = []
        ner_labels = []
        for line in tqdm(lines):
            line = line.replace("\n", "")
            if len(line) == 0:
                lists.append({
                    "id": str(id),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "label": ner_labels
                })
                tokens = []
                ner_tags = []
                ner_labels = []
                id += 1
                continue
            texts = line.split(" ")
            tokens.append(texts[0])
            ner_tags.append(texts[1])
            ner_labels.append(label2id[texts[1]])
    return lists


def main():
    args = NerConfig()

    with open(os.path.join(args.output_dir, "ner_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NerDataset(read_data("../origin/train.txt"),args, tokenizer=tokenizer)
    dev_dataset=NerDataset(read_data("../origin/dev.txt"),args,tokenizer=tokenizer)
    # test_dataset = NerDataset(read_data("../origin/test.txt"), tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=2)

    model = BertNer(args)

    model.to(device)

    train = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        epochs=args.epochs,
        device=device,
        id2label=args.id2label,
        t_total_num=len(train_loader) * args.epochs
    )

    train.train()

    report = train.test()
    print(report)


if __name__ == "__main__":
    main()

    # 预测
    predictor=Predictor()
    texts = [
        "川珍浓香型香菇干货去根肉厚干香菇500g热销品质抢购",
        "女士驾驶证套,复古女行驶证件包驾照夹本卡包驾驶证皮套",
        "17年3月生产宠物体内驱虫打虫,肠虫宁咀嚼片1盒4片",
        "海灵,路利特,卢立康唑乳膏,1%*5g*1支/盒",
        "2017秋冬新款广场舞服装民族风江南古典舞云袖跳舞衣网纱阔腿裤女",
        "蒙王,槟榔十三味丸(高尤&mdash",
        "台湾阿舍食堂台南干面油葱味95g,台湾特色美食特价2018.03.02到期",
        "素面功夫竹骨一尺单面,舞蹈武术团体表演响扇黄蓝黑红太极图",
        "宜客莱,头戴式虚拟现实vr眼镜苹果安卓手机游戏智能3d高清影院"
    ]
    # for text in texts:
    text = texts[0]
    ner_result = predictor.ner_predict(text)
    print("文本>>>>>：", text)
    print("实体>>>>>：", ner_result)
    print("=" * 100)
