"""
This code converts MS MARCO train, dev and eval tsv data into the tfrecord files
that will be consumed by BERT.
"""
import collections
import os
import sys
import re
import tensorflow as tf
import time
# local module
# import tokenization
from transformers import BertTokenizerFast
from tqdm import tqdm, trange
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
logger = logging.getLogger()

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_examples_per_tf", 1000000,
    "The number of examples, which can be divided by 500, to split")

flags.DEFINE_string(
	"output_folder", None,
	"Folder where the tfrecord files will be written.")

flags.DEFINE_string(
	"vocab_file",
	"./data/bert/uncased_L-24_H-1024_A-16/vocab.txt",
	"The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
	"train_dataset_path",
	"./data/triples.train.small.tsv",
	"Path to the MSMARCO training dataset containing the tab separated "
	"<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
	"dev_dataset_path",
	"./data/top1000.dev.tsv",
	"Path to the MSMARCO training dataset containing the tab separated "
	"<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
	"eval_dataset_path",
	"./data/top1000.eval.tsv",
	"Path to the MSMARCO eval dataset containing the tab separated "
	"<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
	"dev_qrels_path",
	"./data/qrels.dev.tsv",
	"Path to the query_id relevant doc ids mapping.")

flags.DEFINE_integer(
	"max_seq_length", 512,
	"The maximum total input sequence length after WordPiece tokenization. "
	"Sequences longer than this will be truncated, and sequences shorter "
	"than this will be padded.")

flags.DEFINE_integer(
	"max_query_length", 64,
	"The maximum query sequence length after WordPiece tokenization. "
	"Sequences longer than this will be truncated.")

flags.DEFINE_integer(
	"num_eval_docs", 1000,
	"The maximum number of docs per query for dev and eval sets.")


def write_to_tf_record(writer, tokenizer, query, docs, labels,
                       ids_file=None, query_id=None, doc_ids=None):
  query = tokenization.convert_to_unicode(query)
  query_token_ids = tokenization.convert_to_bert_input(
      text=query, max_seq_length=FLAGS.max_query_length, tokenizer=tokenizer,
      add_cls=True)

  query_token_ids_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=query_token_ids))

  for i, (doc_text, label) in enumerate(zip(docs, labels)):

    doc_token_id = tokenization.convert_to_bert_input(
          text=tokenization.convert_to_unicode(doc_text),
          max_seq_length=FLAGS.max_seq_length - len(query_token_ids),
          tokenizer=tokenizer,
          add_cls=False)

    doc_ids_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=doc_token_id))

    labels_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[label]))

    features = tf.train.Features(feature={
        'query_ids': query_token_ids_tf,
        'doc_ids': doc_ids_tf,
        'label': labels_tf,
    })
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())

    if ids_file:
     ids_file.write('\t'.join([query_id, doc_ids[i]]) + '\n')


def convert_eval_dataset(set_name, tokenizer):
	def eval_producer():
		pass

	def eval_consumer():
		pass

	print('Converting {} set to tfrecord...'.format(set_name))
	if set_name == 'dev':
		dataset_path = FLAGS.dev_dataset_path
		relevant_pairs = set()
		with open(FLAGS.dev_qrels_path) as f:
			for line in f:
				query_id, _, doc_id, _ = line.strip().split('\t')
				relevant_pairs.add('\t'.join([query_id, doc_id]))
	else:
		dataset_path = FLAGS.eval_dataset_path

	queries_docs = collections.defaultdict(list)
	query_ids = {}
	with open(dataset_path, 'r') as f:
		for i, line in enumerate(f):
			query_id, doc_id, query, doc = line.strip().split('\t')
			label = 0
			if set_name == 'dev':
				if '\t'.join([query_id, doc_id]) in relevant_pairs:
					label = 1
			queries_docs[query].append((doc_id, doc, label))
			query_ids[query] = query_id

	# Add fake paragraphs to the queries that have less than FLAGS.num_eval_docs.
	queries = list(queries_docs.keys())  # Need to copy keys before iterating.
	for query in queries:
		docs = queries_docs[query]
		docs += max(
			0, FLAGS.num_eval_docs - len(docs)) * [('00000000', 'FAKE DOCUMENT', 0)]
		queries_docs[query] = docs

	assert len(
		set(len(docs) == FLAGS.num_eval_docs for docs in queries_docs.values())) == 1, (
		'Not all queries have {} docs'.format(FLAGS.num_eval_docs))

	writers = []
	ids_files = []

	queries = list(query_ids.values())
	num_queries = len(queries_docs)

	print('Counting number of examples...')
	num_lines = sum(1 for line in open(dataset_path, 'r'))
	print('{} examples found.'.format(num_lines))
	num_files = int(num_queries * FLAGS.num_eval_docs / FLAGS.num_examples_per_tf)
	print('num files that will be generated: {}'.format(num_files))
	num_queries_per_file = int(num_queries / num_files)
	print('Num queries per file: {}'.format(num_queries_per_file))

	with ThreadPoolExecutor(max_workers=num_files*2) as executor:
		for k in range(num_files):
			executor.submit(eval_producer(queries[k * num_queries_per_file : (k+1) * num_queries_per_file],
			                              query_ids, queries_docs))
			writer = tf.python_io.TFRecordWriter(
				FLAGS.output_folder + '/dataset_{}_{}.tf'.format(set_name, k))
			writers.append(writer)
			query_doc_ids_path = (
					FLAGS.output_folder + '/query_doc_ids_{}_{}.txt'.format(set_name, k))
			ids_file = open(query_doc_ids_path, 'w')
			ids_files.append(ids_file)
			executor.submit(eval_consumer())

	with open(query_doc_ids_path, 'w') as ids_file:
		for i, (query, doc_ids_docs) in enumerate(queries_docs.items()):
			doc_ids, docs, labels = zip(*doc_ids_docs)
			query_id = query_ids[query]

			write_to_tf_record(writer=writer,
			                   tokenizer=tokenizer,
			                   query=query,
			                   docs=docs,
			                   labels=labels,
			                   ids_file=ids_file,
			                   query_id=query_id,
			                   doc_ids=doc_ids)

			if i % 100 == 0:
				print('Writing {} set, query {} of {}'.format(
					set_name, i, len(queries_docs)))
				time_passed = time.time() - start_time
				hours_remaining = (
						                  len(queries_docs) - i) * time_passed / (max(1.0, i) * 3600)
				print('Estimated hours remaining to write the {} set: {}'.format(
					set_name, hours_remaining))
	dataset_file.close()
	writer.close()


def convert_train_dataset(tokenizer):
	def train_producer(tfrecord_queue, train_dataset_file, lock):
		with lock:
			line = train_dataset_file.readline()
		while line:
			query, positive_doc, negative_doc = line.rstrip().split('\t')
			query_token_ids = tokenizer.tokenize(query, add_special_tokens=True)
			query_token_ids = query_token_ids[:FLAGS.max_query_length]

			query_token_ids_tf = tf.train.Feature(
				int64_list=tf.train.Int64List(value=query_token_ids))

			docs = [positive_doc, negative_doc]
			labels = [1, 0]
			for i, (doc_text, label) in enumerate(zip(docs, labels)):
				doc_token_id = tokenizer.tokenize(doc_text, add_special_tokens=False)
				doc_token_id = doc_token_id[:FLAGS.max_seq_length - len(query_token_ids)]

				doc_ids_tf = tf.train.Feature(
					int64_list=tf.train.Int64List(value=doc_token_id))
				labels_tf = tf.train.Feature(
					int64_list=tf.train.Int64List(value=[label]))

				features = tf.train.Features(feature={
					'query_ids': query_token_ids_tf,
					'doc_ids': doc_ids_tf,
					'label': labels_tf,
				})
				example = tf.train.Example(features=features)
				tfrecord_queue.put(example)
				with lock:
					line = train_dataset_file.readline()
		# Poison pill
		tfrecord_queue.put(None)

	def train_consumer(tfrecord_queue, writer, max_samples_per_file, thread_id):
		example = tfrecord_queue.get()

		for i in tqdm(range(max_samples_per_file), desc="Consumer {}".format(thread_id)):
			if example:
				writer.write(example.SerializeToString())
				example = tfrecord_queue.get()
			else:
				break

	print('Converting to Train to tfrecord...')
	print('Counting number of examples...')
	num_lines = sum(1 for line in open(FLAGS.train_dataset_path, 'r'))
	print('{} examples found.'.format(num_lines))
	num_files = int(num_lines / FLAGS.num_examples_per_tf)
	print('num files that will be generated: {}'.format(num_files))

	tfrecord_queue = Queue()
	train_dataset_file = open(FLAGS.train_dataset_path, 'r')
	read_lock = Lock()

	writers = []
	num_threads = num_files*2
	with ThreadPoolExecutor(max_workers=num_threads) as executor:
		for i in range(num_threads):
			executor.submit(train_producer, tfrecord_queue, train_dataset_file, read_lock)
			writer = tf.python_io.TFRecordWriter(
				FLAGS.output_folder + '/dataset_train_{}.tf'.format(i))
			writers.append(writer)
			executor.submit(train_consumer, tfrecord_queue, writer, num_lines, i)
	tfrecord_queue.join()

	# Close reader and writers
	train_dataset_file.close()
	for writer in writers:
		writer.close()


def main():
	print('Loading Tokenizer...')
	tokenizer = BertTokenizerFast(FLAGS.vocab_file, do_lower_case=True, clean_text=True)

	if not os.path.exists(FLAGS.output_folder):
		os.mkdir(FLAGS.output_folder)

	if FLAGS.train:
		convert_train_dataset(tokenizer=tokenizer)
	elif FLAGS.dev:
		convert_eval_dataset(set_name='dev', tokenizer=tokenizer)
	elif FLAGS.eval:
		convert_eval_dataset(set_name='eval', tokenizer=tokenizer)
	else:
		print("No dataset mode specified!")
	print('Done!')


if __name__ == '__main__':
	main()
	sys.exit(0)
