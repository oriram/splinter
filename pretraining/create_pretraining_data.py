# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""
import functools
import os
import collections
import random
import time
from multiprocessing import Pool

import numpy as np

import tokenization
import tensorflow as tf

from masking import create_geometric_masked_lm_predictions, create_masked_lm_predictions, \
    create_recurring_span_selection_predictions

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_dir", None,
    "Output TF example directory.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_bool("recurring_span_selection", False, "Whether to mask recurring spans")
flags.DEFINE_integer("max_questions_per_seq", 30, "")
flags.DEFINE_bool("only_recurring_span_selection", True, "If set to True, we don't perform MLM")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 80,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer("num_processes", 63, "Number of processes")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float("geometric_masking_p", 0.0, "The p for geometric distribution for span masking.")
flags.DEFINE_integer("max_span_length", 10, "Maximum span length to mask")

flags.DEFINE_bool("verbose", False, "verbose")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, masked_lm_positions, masked_lm_labels,
                 masked_span_positions=None, masked_span_beginnings=None, masked_span_endings=None,
                 input_mask=None):
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.masked_span_positions = masked_span_positions
        self.masked_span_beginnings = masked_span_beginnings
        self.masked_span_endings = masked_span_endings
        self.input_mask = input_mask

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        if self.masked_span_positions is not None:
            s += "masked_span_positions: %s\n" % (" ".join(
                [str(x) for x in self.masked_span_positions]))
            s += "input_mask: %s\n" % (" ".join(
                [str(x) for x in self.input_mask]))
            s += "masked_span_beginnings: %s\n" % (" ".join(
                [str(x) for x in self.masked_span_beginnings]))
            s += "masked_span_endings: %s\n" % (" ".join(
                [str(x) for x in self.masked_span_endings]))

        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class DataStatistics:
    def __init__(self):
        self.num_contexts = 0
        self.ngrams = collections.Counter()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, max_questions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    writer_index = 0

    total_written, total_tokens_written = 0, 0
    for (inst_index, instance) in enumerate(instances):
        # if inst_index % 1000 == 0:
        #     tf.logging.info(f"written {inst_index}")
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_len = len(input_ids)
        input_mask = [1] * input_len if instance.input_mask is None else instance.input_mask
        assert input_len <= max_seq_length
        assert input_len == len(input_mask)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)

        if not FLAGS.only_recurring_span_selection:
            masked_lm_positions = list(instance.masked_lm_positions)
            masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
            masked_lm_weights = [1.0] * len(masked_lm_ids)

            while len(masked_lm_positions) < max_predictions_per_seq:
                masked_lm_positions.append(0)
                masked_lm_ids.append(0)
                masked_lm_weights.append(0.0)

            features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
            features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
            features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        if FLAGS.recurring_span_selection:
            masked_span_positions = list(instance.masked_span_positions)
            masked_span_beginnings = list(instance.masked_span_beginnings)
            masked_span_endings = list(instance.masked_span_endings)
            masked_span_weights = [1.0] * len(masked_span_positions)

            while len(masked_span_positions) < max_questions_per_seq:
                masked_span_positions.append(0)
                masked_span_beginnings.append(0)
                masked_span_endings.append(0)
                masked_span_weights.append(0.0)

            features["masked_span_positions"] = create_int_feature(masked_span_positions)
            features["masked_span_beginnings"] = create_int_feature(masked_span_beginnings)
            features["masked_span_endings"] = create_int_feature(masked_span_endings)
            features["masked_span_weights"] = create_float_feature(masked_span_weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1
        total_tokens_written += input_len

        if FLAGS.verbose and inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info(f"Wrote {total_written} total instances, average length is {total_tokens_written // total_written}")


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_file, tokenizer, max_seq_length, dupe_factor, masked_lm_prob,
                              max_predictions_per_seq, rng, length_dist=None, lengths=None, statistics=None, ngrams=None):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Wiki Input file format:
    # (1) <doc> starts a new wiki document and </doc> ends it
    # (2) One sentence/paragraph per line.
    # BookCorpus file format:
    # Empty line starts a new book
    wiki_format = ("wiki" in input_file)
    with tf.gfile.GFile(input_file, "r") as reader:
        expect_title = False
        for i, line in enumerate(reader):
            # if i % 1000 == 0:
            #     tf.logging.info(f"read {i}")

            line = tokenization.convert_to_unicode(line).strip()
            if wiki_format:
                if (not line) or line.startswith("</doc"):
                    continue

                if expect_title:
                    expect_title = False
                    continue

                # Starting a new document
                if line.startswith("<doc"):
                    all_documents.append([])
                    expect_title = True
                    continue

                tokens = tokenizer.tokenize(line)

            else:
                if not line:
                    all_documents.append([])
                    continue
                line = line.replace('…', "...").replace('’', "'").replace('“', '"').replace('–', "-").replace('—', "-").replace('”', '"').replace('‘', "'")
                tokens = line.split()
                new_tokens = list(tokens)
                for i, t in enumerate(tokens):
                    if t not in tokenizer.vocab:
                        new_tokens[i] = "[UNK]"
                tokens = new_tokens

            if tokens:
                all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for dupe_idx in range(dupe_factor):
        for document_index in range(len(all_documents)):
            # if document_index % 100 == 0:
            #     tf.logging.info(f"processed {document_index}")
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, dupe_idx,
                    length_dist, lengths, statistics, ngrams))

    rng.shuffle(instances)
    return instances


def create_instance_from_context(segments, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, length_dist=None, lengths=None, statistics=None, ngrams=None):
    tokens = ["[CLS]"]
    for segment in segments:
        tokens += segment
    tokens.append("[SEP]")

    if FLAGS.recurring_span_selection:
        tokens, masked_span_positions, input_mask, span_label_beginnings, span_label_endings, span_clusters = \
            create_recurring_span_selection_predictions(tokens,
                                                        FLAGS.max_questions_per_seq,
                                                        FLAGS.max_span_length,
                                                        masked_lm_prob,
                                                        ngrams)
        statistics.ngrams.update([cluster[1] for cluster in span_clusters])
        num_already_masked = len(masked_span_positions)
    else:
        masked_span_positions, input_mask, span_label_beginnings, span_label_endings = None, None, None, None
        num_already_masked = 0

    if FLAGS.only_recurring_span_selection:
        masked_lm_positions, masked_lm_labels = None, None
    elif FLAGS.geometric_masking_p > 0:
        tokens, masked_lm_positions, masked_lm_labels = \
            create_geometric_masked_lm_predictions(tokens, masked_lm_prob, length_dist, lengths, num_already_masked,
                                                   max_predictions_per_seq, vocab_words, rng, input_mask)
    else:
        tokens, masked_lm_positions, masked_lm_labels = \
            create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, num_already_masked,
                                         vocab_words, rng, do_whole_word_mask=FLAGS.do_whole_word_mask)

    statistics.num_contexts += 1

    return TrainingInstance(tokens=tokens,
                            masked_lm_positions=masked_lm_positions,
                            masked_lm_labels=masked_lm_labels,
                            masked_span_positions=masked_span_positions,
                            masked_span_beginnings=span_label_beginnings,
                            masked_span_endings=span_label_endings,
                            input_mask=input_mask)


def create_instances_from_document(
        all_documents, document_index, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_words, rng,
        dupe_idx, length_dist=None, lengths=None, statistics=None, ngrams=None):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    instances = []
    current_chunk = []
    current_length = 0
    for i, segment in enumerate(document):
        segment_len = len(segment)

        if current_length + segment_len > max_num_tokens or (len(document) >= 10 and i % len(document) == dupe_idx):
            if current_chunk:
                current_chunk.append(segment[:max_num_tokens-current_length])
                instance = create_instance_from_context(current_chunk, masked_lm_prob, max_predictions_per_seq,
                                                        vocab_words, rng, length_dist, lengths, statistics, ngrams)
                instances.append(instance)

            current_chunk, current_length = [], 0
            if segment_len > max_num_tokens:
                # If this segment is too long, take the first max_num_tokens from this segment
                segment = segment[:max_num_tokens]
                # instance = create_instance_from_context([segment], masked_lm_prob,
                #                                         max_predictions_per_seq, vocab_words, rng)
                # instances.append(instance)
                # continue

        current_chunk.append(segment)
        current_length += len(segment)

    if current_chunk:
        instance = create_instance_from_context(current_chunk, masked_lm_prob, max_predictions_per_seq,
                                                vocab_words, rng, length_dist, lengths, statistics, ngrams)
        instances.append(instance)

    return instances


def process_file(input_file, output_file, tokenizer, rng, length_dist=None, lengths=None):
    tf.logging.info(f"*** Started processing file {input_file} ***")

    statistics = DataStatistics()

    instances = create_training_instances(
        input_file, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng, length_dist, lengths, statistics)

    tf.logging.info(f"*** Finished processing {statistics.num_contexts} contexts from file {input_file}, "
                    f"writing to output file {output_file} ***")

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, FLAGS.max_questions_per_seq,
                                    [output_file])

    tf.logging.info(f"*** Finished writing to output file {output_file} ***")
    return statistics


def get_output_file(input_file, output_dir):
    path = os.path.normpath(input_file)
    split = path.split(os.sep)
    dir_and_file = split[-2:]
    return os.path.join(output_dir, '_'.join(dir_and_file) + '.tfrecord')


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    assert not tf.io.gfile.exists(FLAGS.output_dir), "Output directory already exists"
    tf.io.gfile.mkdir(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info(f"*** Reading from {len(input_files)} files ***")

    rng = random.Random(FLAGS.random_seed)

    length_dist, lengths = None, None
    if FLAGS.geometric_masking_p > 0:
        p = FLAGS.geometric_masking_p
        lower, upper = 1, FLAGS.max_span_length
        lengths = list(range(lower, upper + 1))
        length_dist = [p * (1 - p) ** (i - lower) for i in range(lower, upper + 1)] if p >= 0 else None
        length_dist = [x / (sum(length_dist)) for x in length_dist]

    params = [(file, get_output_file(file, FLAGS.output_dir), tokenizer, rng, length_dist, lengths)
              for file in input_files]
    with Pool(FLAGS.num_processes if FLAGS.num_processes else None) as p:
        results = p.starmap(process_file, params)

    total_num_contexts = sum([res.num_contexts for res in results])

    tf.logging.info("Finished writing all files! Aggregating statistics..")
    ngrams = functools.reduce(lambda x, y: x + y, [res.ngrams for res in results])
    ngrams = collections.Counter({" ".join(tokens): num for tokens, num in ngrams.items()}).most_common()
    tf.logging.info(f"100 most common ngrams: {ngrams[:100]}")

    ngrams_file = os.path.join(FLAGS.output_dir, "ngrams.txt")
    with tf.gfile.GFile(ngrams_file, "w") as writer:
        for ngram, num in ngrams:
            writer.write(f"{ngram}\t{num}\n")
    tf.logging.info(f"Number of unique n-grams: {len(ngrams)}")

    tf.logging.info(f"Done! Total number of contexts: {total_num_contexts}")


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
