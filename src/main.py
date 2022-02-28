import json
import logging
import os
from argparse import Namespace

import click
import torch, gc
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
torch.cuda.empty_cache()
from tqdm import tqdm
from transformers import WEIGHTS_NAME
from luke.utils.entity_vocab import MASK_TOKEN
from utilis import set_seed
from trainer import Trainer, trainer_args
from model import EntityTyping, CircuitMPE
from data_processor import ENTITY_TOKEN, convert_examples_to_features, DatasetProcessor
from pathlib import Path
here = Path(__file__).parent

logger = logging.getLogger(__name__)


@click.group(name="entity-typing")
def cli():
    pass

@cli.command()
@click.option("--checkpoint-file", type=click.Path(exists=True))
@click.option("--data-dir", default="../../../Phd/datasets/entity_typing/ontonotes_modified", type=click.Path(exists=True))
@click.option("--do-eval/--no-eval", default=True)
@click.option("--do-train/--no-train", default=True)
@click.option("--eval-batch-size", default=32)
@click.option("--num-train-epochs", default=5.0)
@click.option("--seed", default=12)
@click.option("--train-batch-size", default=2)
@trainer_args
@click.pass_obj


def run(common_args, **task_args):
    task_args.update(common_args)
    args = Namespace(**task_args)

    set_seed(args.seed)

    args.experiment.log_parameters({p.name: getattr(args, p.name) for p in run.params})

    args.model_config.vocab_size += 1
    word_emb = args.model_weights["embeddings.word_embeddings.weight"]
    marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
    args.model_weights["embeddings.word_embeddings.weight"] = torch.cat([word_emb, marker_emb])
    args.tokenizer.add_special_tokens(dict(additional_special_tokens=[ENTITY_TOKEN]))

    entity_emb = args.model_weights["entity_embeddings.entity_embeddings.weight"]
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)
    args.model_config.entity_vocab_size = 2
    args.model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])

    train_dataloader, _, features, _ = load_examples(args, fold="train")
    num_labels = len(features[0].labels)

    results = {}

    if args.do_train:
        model = EntityTyping(args, num_labels, is_sdd = True)
        model.load_state_dict(args.model_weights, strict=False)
        model.to(args.device)

        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

        best_dev_average = [-1]
        best_weights = [None]

        def step_callback(model, global_step, writer):
            if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
                epoch = int(global_step / num_train_steps_per_epoch - 1)
                dev_results = evaluate(args, model, fold="dev")
                writer.add_validation_scalar('eval_acc_', dev_results['strict_acc'], epoch)
                args.experiment.log_metrics({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()}, epoch=epoch)
                results.update({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()})
                tqdm.write("dev: " + str(dev_results))

                if dev_results["average"] > best_dev_average[0]:
                    if hasattr(model, "module"):
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in model.module.state_dict().items()}
                    else:
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                    best_dev_average[0] = dev_results["average"]
                    results["best_epoch"] = epoch

                model.train()

        trainer = Trainer(
            args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps, step_callback=step_callback
        )
        trainer.train()

    if args.do_train and args.local_rank in (0, -1):
        logger.info("Saving the model checkpoint to %s", args.output_dir)
        torch.save(best_weights[0], os.path.join(args.output_dir, WEIGHTS_NAME))

    if args.local_rank not in (0, -1):
        return {}

    model = None
    gc.collect()
    torch.cuda.empty_cache()


    if args.do_eval:
        model = EntityTyping(args, num_labels, is_sdd = True)
        if args.checkpoint_file:
            model.load_state_dict(torch.load(args.checkpoint_file, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)

        for eval_set in ("dev", "test"):
            output_file = os.path.join(args.output_dir, f"{eval_set}_predictions.jsonl")
            results.update({f"{eval_set}_{k}": v for k, v in evaluate(args, model, eval_set, output_file).items()})

    logger.info("Results: %s", json.dumps(results, indent=2, sort_keys=True))
    args.experiment.log_metrics(results)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)
        
    return results


def f1(p,r):
    if r == 0.:
        return 0.
    return 2 * p * r / float( p + r )

def strict(true_and_prediction):
    num_entities = len(true_and_prediction)
    correct_num = 0.
    for true_labels, predicted_labels in true_and_prediction:
        correct_num += set(true_labels) == set(predicted_labels)
    try:
        precision = recall = correct_num / num_entities 
    except ZeroDivisionError:
        return 0, 0, 0
    return precision, recall, f1( precision, recall)

def loose_macro(true_and_prediction):
    num_entities = len(true_and_prediction)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
    try:
        precision = p / num_entities
        recall = r / num_entities
    except ZeroDivisionError:
        return 0, 0, 0
    return precision, recall, f1( precision, recall)

def loose_micro(true_and_prediction):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in true_and_prediction:
        # print("true in micro:", true_labels)
        # print("prediction in predicted:",predicted_labels )
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels))) 
    try:
        precision = num_correct_labels / num_predicted_labels
        recall = num_correct_labels / num_true_labels
    except ZeroDivisionError:
        return 0, 0, 0
    return precision, recall, f1( precision, recall)


def evaluate(args, model, fold="dev", output_file=None):
    dataloader, _, _, label_list = load_examples(args, fold=fold)
    model.eval()


    true_and_predictions = []  
    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc=fold):
        #batch - > dict_keys(['word_ids', 'word_attention_mask', 'word_segment_ids', 'entity_ids', 
        #'entity_attention_mask', 'entity_position_ids', 'entity_segment_ids', 'labels'])
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            logits = model(**inputs)      
        logits = logits.detach().cpu().tolist()
        labels = batch["labels"].to("cpu").tolist()

        all_logits.extend(logits)
        all_labels.extend(labels)



    for every_logits, every_labels in zip(all_logits, all_labels):       
        predictions = []
        labels = []
        for i, v in enumerate(every_logits):
            if v >0:
               predictions.append(label_list[i])
        for i, v in enumerate(every_labels):
            if v >0:
                labels.append(label_list[i])
        print("predictions:", predictions)
        print("labels:", labels)
        # predictions.append(label_list[i] for i, v in enumerate(every_logits) if v > 0)
        # labels.append(label_list[i] for i, v in enumerate(every_labels) if v > 0)
        true_and_predictions.append((labels, predictions))  
        # print((labels, predictions))
        # print(true_and_predictions)


    if output_file:
        with open(output_file, "w") as f:
            for labels, predictions in true_and_predictions:
                data = dict(
                    predictions=predictions,
                    labels=labels,
                )
                f.write(json.dumps(data) + "\n")

    micro_f1, macro_f1, acc = loose_micro(true_and_predictions)[2], \
                loose_macro(true_and_predictions)[2], \
                strict(true_and_predictions)[2]

    
    average = (micro_f1+macro_f1+acc)/3


        

    return dict(micro_f1=micro_f1, macro_f1=macro_f1, strict_acc =acc, average = average)


def load_examples(args, fold="train"):
    if args.local_rank not in (-1, 0) and fold == "train":
        torch.distributed.barrier()

    processor = DatasetProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    label_list = processor.get_label_list(args.data_dir)

    logger.info("Creating features from the dataset...")
    features = convert_examples_to_features(examples, label_list, args.tokenizer, args.max_mention_length)

    if args.local_rank == 0 and fold == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        return dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
            labels=torch.tensor([o.labels for o in batch], dtype=torch.long),
        )

    if fold in ("dev", "test"):
        dataloader = DataLoader(features, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(features, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, label_list





   
