

import copy
import csv
import json
import logging
import os

import torch
from torch.utils.data import TensorDataset
# from args import args
# from utils import init_logger, load_tokenizer, set_seed
from utils import get_label

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, id, text, source, target, expression, multi, labels):
        self.id = id
        self.text = text
        self.source = source
        self.target = target
        self.expression = expression
        self.multi = multi
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict())

class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict())

class RelationExample(object):

    def __init__(self, id, text, label):
        self.id = id
        self.text = text
        self.label = label


    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict())

class RelationFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class RelationProcessor(object):
    @classmethod
    def __init__(self, args):
        self.args = args
        self.relation_labels = [1, 0]

    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = int(line[0])
            if i % 1000 == 0:
                logger.info(line)
            examples.append(RelationExample(id=guid, text=text_a, label=label))
        return examples

    def get_examples(self, mode, ling):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        file_dir = os.path.join(self.args.data_dir, ling)

        logger.info("LOOKING AT {}".format(os.path.join(file_dir, file_to_read)))
        return self._create_examples(self._read_tsv(os.path.join(file_dir, file_to_read)), mode)


class ExtractionProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)
        self.label2idx = {"O": 0,
                          "B-targ-Positive": 1,
                          "B-targ-Negative": 1,
                          "B-targ-Neutral": 1,
                          "B-targ-None": 1,
                          "I-targ-Positive": 2,
                          "I-targ-Negative": 2,
                          "I-targ-Neutral": 2,
                          "I-targ-None": 2,
                          "B-exp-Positive": 3,
                          "B-exp-Negative": 4,
                          "B-exp-Neutral": 5,
                          "B-exp-None": 5,
                          "I-exp-Positive": 6,
                          "I-exp-Negative": 7,
                          "I-exp-Neutral": 8,
                          "I-exp-None": 8,
                          "B-holder": 9,
                          "I-holder": 10
                          }

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file) as f:
            lines = []
            for line in f:
                tmp = json.loads(line)
                lines.append(tmp)
        return lines

    def _merge(self, target, expression, source=None):
        res = []
        if source:
            for i in range(len(target)):
                if expression[i] != 'O':
                    res.append(expression[i])
                elif target[i] != 'O':
                    res.append(target[i])
                elif source[i] != 'O':
                    res.append(source[i])
                else:
                    res.append("O")
        else:
            for i in range(len(target)):
                if expression[i] != 'O':
                    res.append(expression[i])
                elif target[i] != 'O':
                    res.append(target[i])
                else:
                    res.append("O")
        return res
    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            id = line['sent_id']
            text = line['text']
            sources = line['sources']
            targets = line['targets']
            expressions = line['expressions']
            multi = self._merge(targets, expressions, sources)
            labels = [self.label2idx[i] for i in multi]
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(id=id, text=text, source=sources, target=targets, expression=expressions, multi=multi, labels=labels))
        return examples

    def get_examples(self, mode, ling):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        file_dir = os.path.join(self.args.data_dir, ling)

        logger.info("LOOKING AT {}".format(os.path.join(file_dir, file_to_read)))
        return self._create_examples(self._read_json(os.path.join(file_dir, file_to_read)))

processors = {'extraction': ExtractionProcessor, 'relation': RelationProcessor}

def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    add_sep_token=False,
    mask_padding_with_zero=True,
):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index == 11:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = tokenizer(example.text, is_split_into_words=True)
        word_ids = tokens.word_ids()
        labels = example.labels

        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(-100)
        # print(new_labels)



        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(tokens["input_ids"])
        input_ids = tokens['input_ids'] + ([pad_token] * padding_length)
        attention_mask = tokens['attention_mask'] + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = tokens['token_type_ids'] + ([pad_token_segment_id] * padding_length)
        labels = new_labels + ([-100] * padding_length)
        #
        #
        #
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        assert len(labels) == max_seq_len, "Error with token type length {} vs {}".format(
            len(labels), max_seq_len
        )
        #
        #
        #
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % example.guid)
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))
        #
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
        )

    return features


def convert_relation_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    add_sep_token=False,
    mask_padding_with_zero=True,
):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index == 3:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text)

        e11_p = [i for i, t in enumerate(tokens_a) if t == '<e1>']  # the start position of entity1
        e12_p = [i for i, t in enumerate(tokens_a) if t == '</e1>']  # the end position of entity1
        e21_p = [i for i, t in enumerate(tokens_a) if t == '<e2>']  # the start position of entity2
        e22_p = [i for i, t in enumerate(tokens_a) if t == '</e2>']  # the end position of entity2

        # Replace the token
        def replace_special_tokens(tokens_a, e_p, flg):
            for i, e in enumerate(e_p):
                tokens_a[e] = flg
                e_p[i] += 1

        replace_special_tokens(tokens_a, e11_p, "$")
        replace_special_tokens(tokens_a, e12_p, "$")
        replace_special_tokens(tokens_a, e21_p, "#")
        replace_special_tokens(tokens_a, e22_p, "#")



        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        if len(tokens_a) > max_seq_len - special_tokens_count:
            tokens_a = tokens_a[: (max_seq_len - special_tokens_count)]

        tokens = tokens_a
        if add_sep_token:
            tokens += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for i, j in zip(e11_p, e12_p):
            e1_mask[i: j + 1] = [1] * (j - i + 1)
        for i, j in zip(e21_p, e22_p):
            e2_mask[i: j + 1] = [1] * (j - i + 1)


        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )

        label_id = int(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.id)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

        features.append(
            RelationFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id,
                e1_mask=e1_mask,
                e2_mask=e2_mask,
            )
        )

    return features

def load_and_cache_relation_examples(args, tokenizer, mode, ling):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        ling,
        "cached_{}_{}_{}_{}".format(
            ling,
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train", ling)
        elif mode == "dev":
            examples = processor.get_examples("dev", ling)
        elif mode == "test":
            examples = processor.get_examples("test", ling)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_relation_examples_to_features(
            examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_label_ids,
        all_e1_mask,
        all_e2_mask,
    )
    return dataset

def load_and_cache_examples(args, tokenizer, mode, ling):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        ling,
        "cached_{}_{}_{}_{}_{}".format(
            ling,
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train", ling)
        elif mode == "dev":
            examples = processor.get_examples("dev", ling)
        elif mode == "test":
            examples = processor.get_examples("test", ling)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.labels for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_label_ids,
    )
    return dataset
