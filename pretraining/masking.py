import time
from collections import namedtuple, defaultdict
import numpy as np
import tensorflow as tf

from tokenization import SPECIAL_TOKENS

STOPWORDS = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'also', 'could', 'would'}

MaskedLmInstance = namedtuple("MaskedLmInstance",
                              ["index", "label"])
MaskedSpanInstance = namedtuple("MaskedSpanInstance",
                              ["index", "begin_label", "end_label"])


def mask_tokens(output_tokens, start_index, end_index, vocab_words, rng):
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
        for idx in range(start_index, end_index+1):
            output_tokens[idx] = "[MASK]"
    else:
        # 10% of the time, replace with random word
        if rng.random() < 0.5:
            for idx in range(start_index, end_index + 1):
                output_tokens[idx] = vocab_words[rng.randint(0, len(vocab_words) - 1)]


def create_geometric_masked_lm_predictions(tokens, masked_lm_prob, length_dist, lengths, num_already_masked,
                                           max_predictions_per_seq, vocab_words, rng, input_mask=None):
    """Creates the predictions for geometric objective."""
    output_tokens = list(tokens)

    candidates_for_start, candidates_for_end, candidates_for_mask = \
        [False] * len(output_tokens), [False] * len(output_tokens), [False] * len(output_tokens)
    for i, token in enumerate(output_tokens):
        if (input_mask is None or input_mask[i]) and token not in SPECIAL_TOKENS:
            candidates_for_mask[i] = True
            candidates_for_start[i] = (not token.startswith("##"))
            candidates_for_end[i] = (
                    i == len(output_tokens) - 1 or not output_tokens[i + 1].startswith("##"))
    if sum(candidates_for_start) < 0.5 * len(output_tokens):
        # logger.info("An example with too many OOV words, skipping on geometric masking")
        candidates_for_start = candidates_for_mask
        candidates_for_end = candidates_for_mask

    num_predictions = 0
    num_tokens_to_mask = int(masked_lm_prob * sum(candidates_for_mask))
    num_tokens_to_mask = min(max_predictions_per_seq - num_already_masked, num_tokens_to_mask)

    len_dist = list(length_dist)
    masked_lms = []
    while num_predictions < num_tokens_to_mask:
        span_len_idx = np.random.choice(range(len(len_dist)), p=len_dist)
        span_len = lengths[span_len_idx]
        if num_predictions + span_len <= num_tokens_to_mask:
            num_attempts = 0
            max_attempts = 30
            while num_attempts < max_attempts:
                start_idx = np.random.randint(len(output_tokens) - span_len + 1)
                end_idx = start_idx + span_len - 1
                if candidates_for_start[start_idx] and candidates_for_end[end_idx] \
                        and all(candidates_for_mask[j] for j in range(start_idx, end_idx + 1)):
                    for j in range(start_idx, end_idx + 1):
                        candidates_for_start[j] = False
                        candidates_for_end[j] = False
                        candidates_for_mask[j] = False
                        masked_lms.append(MaskedLmInstance(index=j, label=output_tokens[j]))

                        num_predictions += 1
                    mask_tokens(output_tokens, start_idx, end_idx, vocab_words, rng)
                    break
                num_attempts += 1
            if num_attempts == max_attempts:
                # print(f"Maximum attempts for span length {span_len}. Skipping geometric masking")
                candidates_for_start = candidates_for_mask
                candidates_for_end = candidates_for_mask

    assert len(masked_lms) <= num_tokens_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, num_already_masked,
                                 vocab_words, rng, do_whole_word_mask=False):
    """Creates the predictions for the masked LM objective."""

    assert 0 < len(tokens) <= 512, str(len(tokens))

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in SPECIAL_TOKENS:
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))
    num_to_predict -= num_already_masked

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            mask_tokens(output_tokens, index, index, vocab_words, rng)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def _iterate_span_indices(span):
    return range(span[0], span[1] + 1)


def get_candidate_span_clusters(tokens, max_span_length, include_sub_clusters=False, validate=True):
    token_to_indices = defaultdict(list)
    for i, token in enumerate(tokens):
        token_to_indices[token].append(i)

    recurring_spans = []
    for token, indices in token_to_indices.items():
        for i, idx1 in enumerate(indices):
            for j in range(i + 1, len(indices)):
                idx2 = indices[j]
                assert idx1 < idx2

                max_recurring_length = 1
                for length in range(1, max_span_length):
                    if include_sub_clusters:
                        recurring_spans.append((idx1, idx2, length))
                    if (idx2 + length) >= len(tokens) or tokens[idx1 + length] != tokens[idx2 + length]:
                        break
                    max_recurring_length += 1

                if max_recurring_length == max_span_length or not include_sub_clusters:
                    recurring_spans.append((idx1, idx2, max_recurring_length))

    spans_to_clusters = {}
    spans_to_representatives = {}
    for idx1, idx2, length in recurring_spans:
        first_span, second_span = (idx1, idx1 + length - 1), (idx2, idx2 + length - 1)
        if first_span in spans_to_representatives:
            if second_span not in spans_to_representatives:
                rep = spans_to_representatives[first_span]
                cluster = spans_to_clusters[rep]
                cluster.append(second_span)
                spans_to_representatives[second_span] = rep
        else:
            cluster = [first_span, second_span]
            spans_to_representatives[first_span] = first_span
            spans_to_representatives[second_span] = first_span
            spans_to_clusters[first_span] = cluster

    if validate:
        recurring_spans = [cluster for cluster in spans_to_clusters.values()
                           if validate_ngram(tokens, cluster[0][0], cluster[0][1] - cluster[0][0] + 1)]
    else:
        recurring_spans = spans_to_clusters.values()
    return recurring_spans


def validate_ngram(tokens, start_index, length):
    # If the vocab at the beginning of the span is a part-of-word (##), we don't want to consider this span.
    # if vocab_word_piece[token_ids[start_index]]:
    if tokens[start_index].startswith("##"):
        return False

    # If the token *after* this considered span is a part-of-word (##), we don't want to consider this span.
    if (start_index + length) < len(tokens) and tokens[start_index + length].startswith("##"):
        return False

    if any([(not tokens[idx].isalnum()) and (not tokens[idx].startswith("##")) for idx in range(start_index, start_index+length)]):
        return False

    # We filter out n-grams that are all stopwords (e.g. "in the", "with my", ...)
    if any([tokens[idx].lower() not in STOPWORDS for idx in range(start_index, start_index+length)]):
        return True
    return False


def get_span_clusters_by_length(span_clusters, seq_length):
    already_taken = [False] * seq_length
    span_clusters = sorted([(cluster, cluster[0][1] - cluster[0][0] + 1) for cluster in span_clusters],
                           key=lambda x: x[1], reverse=True)
    filtered_span_clusters = []
    for span_cluster, _ in span_clusters:
        unpruned_spans = []
        for span in span_cluster:
            if any((already_taken[i] for i in range(span[0], span[1]+1))):
                continue
            unpruned_spans.append(span)

        # Validating that the cluster is indeed "recurring" after the pruning
        if len(unpruned_spans) >= 2:
            filtered_span_clusters.append(unpruned_spans)
            for span in unpruned_spans:
                for idx in _iterate_span_indices(span):
                    already_taken[idx] = True

    return filtered_span_clusters


def create_recurring_span_selection_predictions(tokens, max_recurring_predictions, max_span_length, masked_lm_prob, ngrams=None):
    masked_spans = []
    num_predictions = 0
    input_mask = [1] * len(tokens)
    new_tokens = list(tokens)

    already_masked_tokens = [False] * len(new_tokens)
    span_label_tokens = [False] * len(new_tokens)

    num_to_predict = min(max_recurring_predictions,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    # start_time = time.time()
    span_clusters = get_candidate_span_clusters(tokens, max_span_length, include_sub_clusters=True)
    span_clusters = get_span_clusters_by_length(span_clusters, len(tokens))
    span_clusters = [(cluster, tuple(tokens[cluster[0][0]:cluster[0][1]+1])) for cluster in span_clusters]
    # end_time = time.time()
    # tf.logging.info(f"Finding recurrent ngrams took {end_time - start_time} seconds, {len(tokens)} tokens")

    span_cluster_indices = np.random.permutation(range(len(span_clusters)))
    span_counter = 0
    while span_counter < len(span_cluster_indices):
        span_idx = span_cluster_indices[span_counter]
        span_cluster = span_clusters[span_idx][0]
        # self._assert_and_return_identical(token_ids, identical_spans)
        num_occurrences = len(span_cluster)

        unmasked_span_idx = np.random.randint(num_occurrences)
        unmasked_span = span_cluster[unmasked_span_idx]
        span_counter += 1
        if any([already_masked_tokens[i] for i in _iterate_span_indices(unmasked_span)]):
            # The same token can't be both masked for one pair and unmasked for another pair
            continue

        unmasked_span_beginning, unmasked_span_ending = unmasked_span
        for i, span in enumerate(span_cluster):
            if num_predictions >= num_to_predict:
                # logger.warning(f"Already masked {self.max_predictions} spans.")
                break

            if any([already_masked_tokens[j] for j in _iterate_span_indices(unmasked_span)]):
                break

            if i != unmasked_span_idx:
                if any([already_masked_tokens[j] or span_label_tokens[j] for j in _iterate_span_indices(span)]):
                    # The same token can't be both masked for one pair and unmasked for another pair,
                    # or alternatively masked twice
                    continue

                if any([new_tokens[j] != new_tokens[k] for j, k in
                                       zip(_iterate_span_indices(span), _iterate_span_indices(unmasked_span))]):
                    tf.logging.warning(
                        f"Two non-identical spans: unmasked {new_tokens[unmasked_span_beginning:unmasked_span_ending + 1]}, "
                        f"masked:{new_tokens[span[0]:span[1] + 1]}")
                    continue

                is_first_token = True
                for j in _iterate_span_indices(span):
                    if is_first_token:
                        new_tokens[j] = "[QUESTION]"
                        masked_spans.append(MaskedSpanInstance(index=j,
                                                               begin_label=unmasked_span_beginning,
                                                               end_label=unmasked_span_ending))
                        num_predictions += 1
                    else:
                        new_tokens[j] = "[PAD]"
                        input_mask[j] = 0

                    is_first_token = False
                    already_masked_tokens[j] = True

                for j in _iterate_span_indices(unmasked_span):
                    span_label_tokens[j] = True

    assert len(masked_spans) <= num_to_predict
    masked_spans = sorted(masked_spans, key=lambda x: x.index)

    masked_span_positions = []
    span_label_beginnings = []
    span_label_endings = []
    for p in masked_spans:
        masked_span_positions.append(p.index)
        span_label_beginnings.append(p.begin_label)
        span_label_endings.append(p.end_label)

    return new_tokens, masked_span_positions, input_mask, span_label_beginnings, span_label_endings, span_clusters
