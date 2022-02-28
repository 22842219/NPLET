import json
import os
import jsonlines

from tqdm import tqdm
from transformers import RobertaTokenizer

ENTITY_TOKEN = "[ENTITY]"


class InputExample(object):
    def __init__(self, id_, text, span, labels):
        self.id = id_
        self.text = text
        self.span = span
        self.labels = labels


class InputFeatures(object):
    def __init__(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        labels,
    ):
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.labels = labels


class DatasetProcessor(object):
    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(data_dir, "test")



    def get_label_list(self, data_dir):
        labels = set()
        for split in ['train', 'dev', 'test']:
            examples = self._create_examples(data_dir, split)
            for example in examples:
                labels.update(example.labels)

        return sorted(labels)


    def _create_examples(self, data_dir, set_type):
        with jsonlines.open(os.path.join(data_dir, set_type + ".json"), "r") as f:     
            return [
                InputExample(i+j, ' '.join(line["tokens"]), (mention["start"], mention["end"]), mention["labels"]) \
                    for i, line in enumerate(f) \
                    for j, mention in enumerate(line["mentions"]) 
            ]

# #    ZIYU: FOR DATA ANAYLYSIS
#     def _create_examples(self, data_dir, set_type):
#         with jsonlines.open(os.path.join(data_dir, set_type + ".json"), "r") as f:     
#             return [
#                 InputExample(i+j, ' '.join(line["tokens"]), (mention["start"], mention["end"]), mention["labels"]) \
#                     for i, line in enumerate(f) \
#                     for j, mention in enumerate(line["mentions"]) \
#                     if '/PERSON' in mention['labels'] 
#             ]


 
    def _extract_examples(self, data_dir, set_type):
        labels = []
        with jsonlines.open(os.path.join(data_dir, set_type + ".json"), "r") as f:     
            for i, line in enumerate(f):
                for j, mention in enumerate(line["mentions"]):
                    if '/PERSON' in mention['labels']:
                        # print(line)
                        for label in mention['labels']:
                            if label not in labels:
                                labels.append(label)
        return labels
    
    def extract_train_examples(self, data_dir):
        return self._extract_examples(data_dir, "train")

    def extract_dev_examples(self, data_dir):
        return self._extract_examples(data_dir, "dev")

    def extract_test_examples(self, data_dir):
        return self._extract_examples(data_dir, "test")


def convert_examples_to_features(examples, label_list, tokenizer, max_mention_length):
    label_map = {label: i for i, label in enumerate(label_list)}

    conv_tables = (
        ("-LRB-", "("),
        ("-LCB-", "("),
        ("-LSB-", "("),
        ("-RRB-", ")"),
        ("-RCB-", ")"),
        ("-RSB-", ")"),
    )
    features = []
    for example in tqdm(examples):

        def preprocess_and_tokenize(text, start, end=None):
            target_text = text[start:end]
            for a, b in conv_tables:
                target_text = target_text.replace(a, b)

            if isinstance(tokenizer, RobertaTokenizer):
                return tokenizer.tokenize(target_text, add_prefix_space=True)
            else:
                return tokenizer.tokenize(target_text)

        tokens = [tokenizer.cls_token]
        tokens += preprocess_and_tokenize(example.text, 0, example.span[0])
        mention_start = len(tokens)
        tokens.append(ENTITY_TOKEN)
        tokens += preprocess_and_tokenize(example.text, example.span[0], example.span[1])
        tokens.append(ENTITY_TOKEN)
        mention_end = len(tokens)

        tokens += preprocess_and_tokenize(example.text, example.span[1])
        tokens.append(tokenizer.sep_token)

        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        word_attention_mask = [1] * len(tokens)
        word_segment_ids = [0] * len(tokens)

        entity_ids = [1, 0]
        entity_attention_mask = [1, 0]
        entity_segment_ids = [0, 0]
        entity_position_ids = list(range(mention_start, mention_end))[:max_mention_length]
        entity_position_ids += [-1] * (max_mention_length - mention_end + mention_start)
        entity_position_ids = [entity_position_ids, [-1] * max_mention_length]

        labels = [0] * len(label_map)

        for label in example.labels:
            labels[label_map[label]] = 1

        features.append(
            InputFeatures(
                word_ids=word_ids,
                word_segment_ids=word_segment_ids,
                word_attention_mask=word_attention_mask,
                entity_ids=entity_ids,
                entity_position_ids=entity_position_ids,
                entity_segment_ids=entity_segment_ids,
                entity_attention_mask=entity_attention_mask,
                labels=labels,
            )
        )

    return features





def main():
    import os
    import json, jsonlines

    import logging
    logger = logging.getLogger(__name__)

    # from transformers import LukeTokenizer, LukeForEntityClassification
    # tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
    # max_mention_length = 4


    import configparser
    config = configparser.ConfigParser()

    config.read('../../config.ini')
    filenames = config["FILENAMES"]
    input_data_folder = filenames['et_folder']

    bbn_modified  = os.path.join(input_data_folder, 'bbn_modified')
    print(bbn_modified)


    processor = DatasetProcessor()
  

    train_examples = processor.extract_train_examples(bbn_modified)
    dev_examples = processor.extract_dev_examples(bbn_modified)
    test_examples = processor.extract_test_examples(bbn_modified)

    print(train_examples)
    print(dev_examples)
    print(test_examples)
    # labels = set()
    # for split in ['train', 'dev', 'test']:
    #     examples = '{}_examples'.format(split)
    #     for example in examples:
    #         labels.update(example.labels)

    # print(sorted(labels))


    # label_list = processor.get_label_list(bbn_modified)
    # print(label_list, len(label_list))
    

    logger.info("Creating features from the dataset...")
    # features = convert_examples_to_features(train_examples, label_list, tokenizer, max_mention_length)
    
  



if __name__ == "__main__":
    main()
