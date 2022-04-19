from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import os
import logging
import numpy as np
from main_code.model.agent import EntityAgent
from main_code.options import read_options
from main_code.model.environment import env
import codecs
from collections import defaultdict
import gc
import resource
import sys
from main_code.model.baseline import ReactiveBaseline
from scipy.special import logsumexp as lse
import torch
import torch.optim as optim
from main_code.model.nell_eval import nell_eval
import random


logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

class Trainer(object):
	def __init__(self, params):

		# transfer parameters to self
		for key, val in params.items(): setattr(self, key, val);
		self.device = params['device']
		print(self.device)
		self.agent = EntityAgent(params).to(self.device)

		self.save_path = self.model_dir + "model" + '.ckpt'

		self.train_environment = env(params, 'train')
		self.dev_test_environment = env(params, 'dev')
		self.test_test_environment = env(params, 'test')
		self.test_environment = self.dev_test_environment

		self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
		self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab

		self.max_hits_at_10 = 0
		self.ePAD = self.entity_vocab['PAD']
		self.rPAD = self.relation_vocab['PAD']

		self.decaying_beta_init = self.beta
		# optimize
		self.baseline = ReactiveBaseline(params, self.Lambda)

		self.positive_reward_rates = []

		self.optimizer = optim.Adam(list(self.agent.parameters()), lr=self.learning_rate)

	def calc_reinforce_loss(self, all_loss, all_logits, cum_discounted_reward, decaying_beta, baseline, embedding_reward= None):

		loss = torch.stack(all_loss, dim=1)  # [original batch_size * num_rollout, T]
		base_value = baseline.get_baseline_value()

		# multiply with rewards
		final_reward = cum_discounted_reward - base_value

		reward_mean = torch.mean(final_reward)

		# Constant added for numerical stability
		reward_std = torch.std(final_reward) + 1e-6
		final_reward = torch.div(final_reward - reward_mean, reward_std)

		loss = torch.mul(loss, final_reward)  # [original batch_size * num_rollout, T]
		entropy_loss = decaying_beta * self.entropy_reg_loss(all_logits)
		total_loss = torch.mean(loss) - entropy_loss  # scalar

		return total_loss

	def entropy_reg_loss(self, all_logits):  # control diversity
		all_logits = torch.stack(all_logits, dim=2)  # [original batch_size * num_rollout, max_num_actions, T]
		entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))  # scalar

		return entropy_loss

	"""
	computed cumulative discounted reward
	"""
	def calc_cum_discounted_reward(self, rewards):
		running_add = torch.zeros([rewards.size(0)]).to(self.device)  # [original batch_size * num_rollout]
		cum_disc_reward = torch.zeros([rewards.size(0), self.path_length]).to(self.device)  # [original batch_size * num_rollout, T]
		cum_disc_reward[:, self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
		for t in reversed(range(self.path_length)):
			running_add = self.gamma * running_add + cum_disc_reward[:, t]
			cum_disc_reward[:, t] = running_add

		return cum_disc_reward

	def train(self):

		logger.info("Begin train\n")
		train_loss = 0.0
		start_time = time.time()
		self.batch_counter = 0
		current_decay = self.decaying_beta_init
		current_decay_count = 0

		# training epoch
		for entity_episode in self.train_environment.get_episodes():

			self.batch_counter += 1

			current_decay_count += 1
			if current_decay_count == self.decay_batch:
				current_decay *= self.decay_rate
				current_decay_count = 0

			# get initial state for entity agent
			# LSTM(h_0, c_0)
			state_emb = torch.zeros(2, self.batch_size * self.num_rollouts, self.agent.m * self.embedding_size).to(self.device)
			state_emb = (state_emb[0], state_emb[1])

			# state (e_next, r_next, l_next, e_current)
			state = entity_episode.get_state()

			#state init
			next_possible_entities = torch.tensor(state['next_entities']).long().to(self.device)
			next_possible_relations = torch.tensor(state['next_relations']).long().to(self.device)

			# prev state  original batch_size * num_rollout, 1-D, (1...)
			prev_relation = self.agent.dummy_start_label.to(self.device)
			prev_entity = torch.tensor(state['current_entities']).long().to(self.device)

			"""query_relation不知道怎么处理"""
			query_relation = entity_episode.get_query_relation()
			query_relation = torch.tensor(query_relation).long().to(self.device)

			first_step_of_test = False

			range_arr = torch.arange(self.batch_size * self.num_rollouts).to(self.device)

			# 统计结果初始化
			all_losses = []
			all_logits = []
			all_action_id = []


			# head_entities = get_value(self.entity_vocab_change, prev_entity.cpu().numpy())

			# reasoning step
			for i in range(self.path_length):
				loss, state_emb, logits, idx, chosen_relation = self.agent.step(next_possible_relations,
																								next_possible_entities,
																								state_emb,
																								prev_relation,
																								query_relation,
																								prev_entity,
																								range_arr,
																								first_step_of_test
																								)

				new_state = entity_episode(idx)  ## important !! switch to next state with new entity and new relation

				# the new entity
				next_possible_entities = torch.tensor(new_state['next_entities']).long().to(self.device)

				# prev_state
				prev_entity = torch.tensor(new_state['current_entities']).long().to(self.device)
				prev_relation = chosen_relation.to(self.device)

				# all_loss
				all_losses.append(loss)
				all_logits.append(logits)
				all_action_id.append(idx)

				# embedding reward
				# query assignment

				# fill_query and get embedding reward


			# get the final reward from the environment
			rewards = entity_episode.get_reward()
			rewards_torch = torch.tensor(rewards).to(self.device)
			cum_discounted_reward = self.calc_cum_discounted_reward(rewards_torch)


			reinforce_loss = self.calc_reinforce_loss(all_losses, all_logits, cum_discounted_reward,
													  current_decay, self.baseline)


			self.baseline.update(torch.mean(cum_discounted_reward))

			self.optimizer.zero_grad()
			reinforce_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=self.grad_clip_norm, norm_type=2)
			self.optimizer.step()

			# print statistics
			train_loss = 0.98 * train_loss + 0.02 * reinforce_loss
			avg_reward = np.mean(rewards)
			self.positive_reward_rates.append(avg_reward)

			# now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
			# entity pair, atleast one of the path get to the right answer
			reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
			reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
			reward_reshape = (reward_reshape > 0)
			num_ep_correct = np.sum(reward_reshape)
			if np.isnan(train_loss.item()):
				raise ArithmeticError("Error in computing loss")

			logger.info("Agents: batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
						"num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
						format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
							   (num_ep_correct / self.batch_size),
							   train_loss))

			# if self.batch_counter % self.eval_every == 0:
			if self.batch_counter ==1:
				# self.test_rollouts = 100
				self.test_environment = self.test_test_environment

				with open(self.output_dir + '/scores.txt', 'a') as score_file:
					score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
				os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
				self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

				self.test(beam=True, print_paths=False)

			logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

			gc.collect()
			if self.batch_counter >= self.total_iterations:
				break

		np.save(self.reward_dir, self.positive_reward_rates)

	def test(self, beam=False, print_paths=False, save_model=True, auc=False):

		with torch.no_grad():

			batch_counter = 0
			paths = defaultdict(list)
			answers = []
			all_final_reward_1 = 0
			all_final_reward_3 = 0
			all_final_reward_5 = 0
			all_final_reward_10 = 0
			all_final_reward_20 = 0
			auc = 0

			total_examples = self.test_environment.total_no_examples

			for entity_episode in tqdm(self.test_environment.get_episodes()):
				batch_counter += 1

				temp_batch_size = entity_episode.no_examples

				self.qr = entity_episode.get_query_relation()
				query_relation = self.qr
				query_relation = torch.tensor(query_relation).long().to(self.device)

				# set initial beam probs

				beam_probs = torch.zeros((temp_batch_size * self.test_rollouts, 1)).to(self.device)

				# get initial state for entity agent
				state = entity_episode.get_state()

				next_relations = torch.tensor(state['next_relations']).long().to(self.device)
				next_entities = torch.tensor(state['next_entities']).long().to(self.device)
				current_entities = torch.tensor(state['current_entities']).long().to(self.device)

				state_emb = torch.zeros(2, temp_batch_size * self.test_rollouts, self.agent.m * self.embedding_size).to(self.device)
				state_emb = (state_emb[0], state_emb[1])

				prev_relations = (torch.ones(temp_batch_size * self.test_rollouts) * self.relation_vocab[
					'DUMMY_START_RELATION']).long().to(self.device)

				range_arr = torch.arange(temp_batch_size * self.test_rollouts).to(self.device)

				####logs####
				if print_paths:
					self.entity_trajectory = []
					self.relation_trajectory = []
				####################
				"""
				need repaired
				"""
				self.log_probs = np.zeros((temp_batch_size * self.test_rollouts,)) * 1.0

				# for each time step
				for i in range(self.path_length):
					if i == 0:
						first_state_of_test = True

					loss, state_emb, test_scores, test_action_idx, chosen_relations = self.agent.step(next_relations,
													  					next_entities,
													  					state_emb,
													  					prev_relations,
													  					query_relation,
													  					current_entities,
													  					range_arr,
													  					first_state_of_test
    												  					)

					if beam:
						k = self.test_rollouts
						beam_probs = beam_probs.to(self.device)
						new_scores = test_scores + beam_probs
						new_scores = new_scores.cpu()
						if i == 0:
							# 得分排序
							idx = np.argsort(new_scores)
							idx = idx[:, -k:]
							ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
							idx = idx[np.arange(k * temp_batch_size), ranged_idx]
						else:
							idx = self.top_k(new_scores, k)

						y = idx // (self.max_num_actions)
						x = idx % (self.max_num_actions)

						y += np.repeat([b * k for b in range(temp_batch_size)], k)
						state['current_entities'] = state['current_entities'][y]
						state['next_relations'] = state['next_relations'][y, :]
						state['next_entities'] = state['next_entities'][y, :]
						# state_emb = state_emb[:, :, y, :]
						# state_emb[0] = h_t, state_emb[1] = c_t
						state_emb = (state_emb[0][y, :], state_emb[1][y, :])

						test_action_idx = x
						chosen_relations = state['next_relations'][np.arange(temp_batch_size * k), x]

						beam_probs = new_scores[y, x]
						beam_probs = beam_probs.reshape((-1, 1))
						if print_paths:
							for j in range(i):
								self.entity_trajectory[j] = self.entity_trajectory[j][y]
								self.relation_trajectory[j] = self.relation_trajectory[j][y]


					state = entity_episode(test_action_idx)

					next_relations = torch.tensor(state['next_relations']).long().to(self.device)
					next_entities = torch.tensor(state['next_entities']).long().to(self.device)

					current_entities = torch.tensor(state['current_entities']).long().to(self.device)

					prev_relations = torch.tensor(chosen_relations).long().to(self.device)

					####logs####
					if print_paths:
						self.entity_trajectory.append(state['current_entities'])
						self.relation_trajectory.append(chosen_relations)
					####################
					test_scores = test_scores.cpu().numpy()
					self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx.cpu().numpy()]
				if beam:
					self.log_probs = beam_probs

				####Logs####

				if print_paths:
					self.entity_trajectory.append(state['current_entities'])

				# ask environment for final reward
				rewards = entity_episode.get_reward()
				# [orig_batch, test_rollouts]
				reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))
				self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
				sorted_indx = np.argsort(-self.log_probs)
				final_reward_1 = 0
				final_reward_3 = 0
				final_reward_5 = 0
				final_reward_10 = 0
				final_reward_20 = 0
				AP = 0
				ce = entity_episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
				se = entity_episode.start_entities.reshape((temp_batch_size, self.test_rollouts))
				for b in range(temp_batch_size):
					answer_pos = None
					seen = set()
					pos = 0
					if self.pool == 'max':
						for r in sorted_indx[b]:
							if reward_reshape[b, r] == self.positive_reward:
								answer_pos = pos
								break
							if ce[b, r] not in seen:
								seen.add(ce[b, r])
								pos += 1
					if self.pool == 'sum':
						scores = defaultdict(list)
						answer = ''
						for r in sorted_indx[b]:
							scores[ce[b, r]].append(self.log_probs[b, r])
							if reward_reshape[b, r] == self.positive_reward:
								answer = ce[b, r]
						final_scores = defaultdict(float)
						for e in scores:
							final_scores[e] = lse(scores[e])
						sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
						if answer in sorted_answers:
							answer_pos = sorted_answers.index(answer)
						else:
							answer_pos = None

					if answer_pos != None:
						if answer_pos < 20:
							final_reward_20 += 1
							if answer_pos < 10:
								final_reward_10 += 1
								if answer_pos < 5:
									final_reward_5 += 1
									if answer_pos < 3:
										final_reward_3 += 1
										if answer_pos < 1:
											final_reward_1 += 1
					if answer_pos == None:
						AP += 0
					else:
						AP += 1.0 / ((answer_pos + 1))
					if print_paths:
						qr = self.train_environment.grapher.rev_relation_vocab[self.qr[b * self.test_rollouts]]
						start_e = self.rev_entity_vocab[entity_episode.start_entities[b * self.test_rollouts]]
						end_e = self.rev_entity_vocab[entity_episode.end_entities[b * self.test_rollouts]]
						paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
						paths[str(qr)].append(
							"Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
						for r in sorted_indx[b]:
							indx = b * self.test_rollouts + r
							if rewards[indx] == self.positive_reward:
								rev = 1
							else:
								rev = -1
							answers.append(
								self.rev_entity_vocab[se[b, r]] + '\t' + self.rev_entity_vocab[ce[b, r]] + '\t' + str(
									self.log_probs[b, r].item()) + '\n')
							paths[str(qr)].append(
								'\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in
										   self.entity_trajectory]) + '\n' + '\t'.join(
									[str(self.rev_relation_vocab[re[indx]]) for re in
									 self.relation_trajectory]) + '\n' + str(
									rev) + '\n' + str(
									self.log_probs[b, r]) + '\n___' + '\n')
						paths[str(qr)].append("#####################\n")

				all_final_reward_1 += final_reward_1
				all_final_reward_3 += final_reward_3
				all_final_reward_5 += final_reward_5
				all_final_reward_10 += final_reward_10
				all_final_reward_20 += final_reward_20
				auc += AP

			all_final_reward_1 /= total_examples
			all_final_reward_3 /= total_examples
			all_final_reward_5 /= total_examples
			all_final_reward_10 /= total_examples
			all_final_reward_20 /= total_examples
			auc /= total_examples
			if save_model:
				if all_final_reward_10 >= self.max_hits_at_10:
					self.max_hits_at_10 = all_final_reward_10
					torch.save(self.agent.state_dict(), self.model_dir + "e_model" + '.ckpt')
					# self.save_path = self.model_dir + "model" + '.ckpt'

			if print_paths:
				logger.info("[ printing paths at {} ]".format(self.output_dir + '/test_beam/'))
				for q in paths:
					j = q.replace('/', '-')
					with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
						for p in paths[q]:
							pos_file.write(p)
				with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
					for a in answers:
						answer_file.write(a)

			with open(self.output_dir + '/scores.txt', 'a') as score_file:
				score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
				score_file.write("\n")
				score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
				score_file.write("\n")
				score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
				score_file.write("\n")
				score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
				score_file.write("\n")
				score_file.write("Hits@20: {0:7.4f}".format(all_final_reward_20))
				score_file.write("\n")
				score_file.write("auc: {0:7.4f}".format(auc))
				score_file.write("\n")
				score_file.write("\n")

			logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
			logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
			logger.info("Hits@5: {0:7.4f}".format(all_final_reward_5))
			logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
			logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
			logger.info("auc: {0:7.4f}".format(auc))

	def top_k(self, scores, k):
		# scores = scores.reshape(-1, k * self.max_num_actions)  # [B, (k*max_num_actions)]
		scores = scores.reshape(-1, k * (self.max_num_actions))
		idx = np.argsort(scores, axis=1)
		idx = idx[:, -k:]  # take the last k highest indices # [B , k]
		return idx.reshape((-1))


if __name__ == '__main__':

	# read command line options
	options = read_options()
	set_global_seed(options['seed'])
	# if options['use_embedding']:
	# 	options = read_pretrained_embeddings(options)
	options['device'] = 'cuda' if options['use_cuda'] else 'cpu'

	# Set logging
	logger.setLevel(logging.INFO)
	fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
							'%m/%d/%Y %I:%M:%S %p')
	console = logging.StreamHandler()
	console.setFormatter(fmt)
	logger.addHandler(console)
	logfile = logging.FileHandler(options['log_file_name'], 'w')
	logfile.setFormatter(fmt)
	logger.addHandler(logfile)
	# read the vocab files, it will be used by many classes hence global scope
	logger.info('reading vocab files...')
	options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
	options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))


	logger.info('Reading mid to name map')
	mid_to_word = {}

	logger.info('Done..')
	logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
	logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))

	save_path = ''


	# Training
	if not options['load_model']:
		trainer = Trainer(options)
		trainer.train()
		save_path = trainer.save_path
		path_logger_file = trainer.path_logger_file
		output_dir = trainer.output_dir

	# Testing on test with best model
	else:
		logger.info("Skipping training")
		logger.info("Loading model from {}".format(options["model_load_dir"]))

	# trainer = Trainer(options)
	# if options['load_model']:
	# 	save_path = options['model_load_dir']
	# 	path_logger_file = trainer.path_logger_file
	# 	output_dir = trainer.output_dir
	#
	# trainer.agent.load_state_dict(torch.load(save_path))

	os.mkdir(path_logger_file + "/" + "test_beam")
	trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
	with open(output_dir + '/scores.txt', 'a') as score_file:
		score_file.write("Test (beam) scores with best model from " + save_path + "\n")
	trainer.test_environment = trainer.test_test_environment

	trainer.test(beam=True, print_paths=True, save_model=False)

	print(options['nell_evaluation'])
	if options['nell_evaluation'] == 1:
		nell_eval(path_logger_file + "/" + "test_beam/" + "pathsanswers", trainer.data_input_dir + '/sort_test.pairs')

