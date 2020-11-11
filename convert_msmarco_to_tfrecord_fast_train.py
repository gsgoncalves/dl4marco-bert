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

flags.DEFINE_boolean(
    "train", False,
    "Dataset mode")

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

def convert_train_dataset(tokenizer):
	def train_producer(tfrecord_queue, train_dataset_file, lock):
		#with lock:
		line = train_dataset_file.readline()
		while True:
			if len(line) > 0:
				# query, positive_doc, negative_doc = '#'.join(line.split('#')[:-1]).rstrip().split('\t')
				query, positive_doc, negative_doc = line.rstrip().split('\t')
				query_tokens = tokenizer.tokenize(query, add_special_tokens=True, truncation=True,
				                                  max_length=FLAGS.max_query_length)
				query_token_ids = tokenizer.convert_tokens_to_ids(query_tokens)
				query_token_ids_tf = tf.train.Feature(
					int64_list=tf.train.Int64List(value=query_token_ids))

				docs = [positive_doc, negative_doc]
				labels = [1, 0]
				for i, (doc_text, label) in enumerate(zip(docs, labels)):
					doc_tokens = tokenizer.tokenize(doc_text, add_special_tokens=False, truncation=True,
					                                max_length=FLAGS.max_seq_length-len(query_token_ids)-1)
					doc_tokens += '[SEP]'

					doc_token_id = tokenizer.convert_tokens_to_ids(doc_tokens)

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
					#print("Putting example")
					tfrecord_queue.put(example, timeout=5)
					#print("Done")
					#with lock:
					line = train_dataset_file.readline()
					#print(line)
					#print(len(line))
			else:
				print("POISON PILL")
				# Poison pill
				tfrecord_queue.put(None)
				break

	def train_consumer(tfrecord_queue, writer, max_samples_per_file, thread_id):
		example = tfrecord_queue.get()

		for _ in trange(max_samples_per_file, desc="Consumer {}".format(thread_id)):
			if example is not None:
				writer.write(example.SerializeToString())
				#print("Waiting for the example")
				example = tfrecord_queue.get()
				#print("Got the example")
			else:
				break
		print("Out of the loop")

	print('Converting to Train to tfrecord...')
	print('Counting number of examples...')
	num_lines = sum(1 for line in open(FLAGS.train_dataset_path, 'r'))
	print('{} examples found.'.format(num_lines))
	num_files = max(1, int(num_lines / FLAGS.num_examples_per_tf))
	print('num files that will be generated: {}'.format(num_files))
	num_lines = min(num_lines, FLAGS.num_examples_per_tf)

	tfrecord_queue = Queue()
	train_dataset_file = open(FLAGS.train_dataset_path, 'r')
	read_lock = Lock()

	writers = []
	num_threads = num_files * 2
	with ThreadPoolExecutor(max_workers=num_threads) as executor:
		for i in range(num_files):
			writer = tf.python_io.TFRecordWriter(
				FLAGS.output_folder + '/dataset_train_{}.tf'.format(i))
			writers.append(writer)
			executor.submit(train_consumer, tfrecord_queue, writer, num_lines, i)
			executor.submit(train_producer, tfrecord_queue, train_dataset_file, read_lock)
			# executor.submit(train_consumer, tfrecord_queue, writer, num_lines, i)
	#tfrecord_queue.join()

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
	#elif FLAGS.dev:
	#	convert_eval_dataset(set_name='dev', tokenizer=tokenizer)
	#elif FLAGS.eval:
#		convert_eval_dataset(set_name='eval', tokenizer=tokenizer)
	else:
		print("No dataset mode specified!")
	print('Done!')


if __name__ == '__main__':
	main()
	sys.exit(0)
