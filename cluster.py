#!/usr/bin/python

import argparse
from subprocess import Popen


def preprocess(args):
	cmdline = [
		'srun', '--mem=%d' % args.mem, '-c%d' % args.cpus, '--time=%s' % args.time,
		'--job-name=%s-preprocess' % args.data_name, '--output=out/preprocess-%s.out' % args.data_name,
		'scripts/preprocess.sh', args.data_name, args.dataset, ' '.join(args.speakers), ' '.join(args.noise_dirs),
	]

	print(' '.join(cmdline))
	Popen(cmdline)


def train(args):
	cmdline = [
		'srun', '--mem=%d' % args.mem, '--gres=gpu:%d' % args.gpus, '-c%d' % args.cpus, '--time=%s' % args.time,
		'--job-name=%s-train' % args.model, '--output=out/train-%s.out' % args.model,
		'scripts/train.sh', args.model, ' '.join(args.train_data_names), ' '.join(args.validation_data_names)
	]

	print(' '.join(cmdline))
	Popen(cmdline)


def predict(args):
	cmdline = [
		'srun', '--mem=%d' % args.mem, '--gres=gpu:%d' % args.gpus, '-c%d' % args.cpus, '--time=%s' % args.time,
		'--job-name=%s-predict' % args.model, '--output=out/predict-%s.out' % args.model,
		'scripts/predict.sh', args.model, args.dataset, ' '.join(args.speakers), ' '.join(args.noise_dirs)
	]

	print(' '.join(cmdline))
	Popen(cmdline)


def main():
	parser = argparse.ArgumentParser()

	action_parsers = parser.add_subparsers()

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-dn', '--data_name', type=str, required=True)
	preprocess_parser.add_argument('-ds', '--dataset', type=str, required=True)
	preprocess_parser.add_argument('-s', '--speakers', nargs='+', type=str, required=True)
	preprocess_parser.add_argument('-n', '--noise_dirs', nargs='+', type=str, required=True)
	preprocess_parser.add_argument('-m', '--mem', type=int, default=150000)
	preprocess_parser.add_argument('-c', '--cpus', type=int, default=8)
	preprocess_parser.add_argument('-t', '--time', type=str, default='10:0:0')
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-mn', '--model', type=str, required=True)
	train_parser.add_argument('-tdn', '--train_data_names', nargs='+', type=str, required=True)
	train_parser.add_argument('-vdn', '--validation_data_names', nargs='+', type=str, required=True)
	train_parser.add_argument('-m', '--mem', type=int, default=100000)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.add_argument('-c', '--cpus', type=int, default=4)
	train_parser.add_argument('-t', '--time', type=str, default='30:0:0')
	train_parser.set_defaults(func=train)

	predict_parser = action_parsers.add_parser('predict')
	predict_parser.add_argument('-mn', '--model', type=str, required=True)
	predict_parser.add_argument('-ds', '--dataset', type=str, required=True)
	predict_parser.add_argument('-s', '--speakers', nargs='+', type=str, required=True)
	predict_parser.add_argument('-n', '--noise_dirs', nargs='+', type=str, required=True)
	predict_parser.add_argument('-m', '--mem', type=int, default=50000)
	predict_parser.add_argument('-g', '--gpus', type=int, default=1)
	predict_parser.add_argument('-c', '--cpus', type=int, default=4)
	predict_parser.add_argument('-t', '--time', type=str, default='10:0:0')
	predict_parser.set_defaults(func=predict)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
