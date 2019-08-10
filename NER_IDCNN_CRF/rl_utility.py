# encoding: utf-8
# Copyright 2019 The DeepNlp Authors.
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
# limitations under the License
"""
@file: rl_utility.py
@time: 2019-08-10 22:01
"""

import sys, os


class SequenceLabelRLInterface(object):
    """
    序列识别应用增强学习调优的接口
    1. 我们以目标tag对应的概率为action的概率；选择哪些tag做为action是底层实现需要考虑的因素
    2. reward参考最终的预测输出和目标输出，根据具体的场景来计算奖励

    """
    def get_action_probability(self, hidden_tensor, softmax_tensor, length_tensor, target_tensor):
        """
        获得
        :param hidden_tensor:
        :param softmax_tensor:
        :param target_tensor:
        :return:
        """
        raise NotImplementedError("you must implement action & probability interface")

    def compute_reward(self, output_tags, target_tags):
        """

        :param output_tags: batch_size, length, e.g. [ [O, O, B-LOC, I-LOC, O, O, O], [O, O, B-LOC, I-LOC, O, O], ...]
        :param target_tags: batch_size, length, e.g. [ [B-PER, O-PER, B-LOC, I-LOC, O, O, O],
        [O, O, B-LOC, I-LOC, O, O], ...]
        :return: rewards with shape (batch_size)
        """
        raise NotImplementedError("you must implement reward function")


class EffectiveWholeLabelRL(SequenceLabelRLInterface):
    """
    reward机制， 所有有效的节点都识别到了得1分， 否则得负分。
    如果有1/5的tag没有打上，得-0.2分， 如果一个都没有识别出， 得-1分,
    如果有多识别出来的元素，每多一个扣0.1分

    """

    def __init__(self, effective_tags=('B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER'),
                 corresponding_weights=(1,1,1,1,1,1)):
        """

        :param effective_tags:  有效的识别tag
        :param corresponding_weights: tag对应的权重
        """
        assert(len(effective_tags) == len(corresponding_weights))
        self.effective_tag_dict = dict()
        sum_weights = sum(corresponding_weights)
        for index in range(len(effective_tags)):
            tag_weight =corresponding_weights[index] *1./sum_weights
            tag = effective_tags[index]
            self.effective_tag_dict[tag] = tag_weight
        self.other_wrong_score = -0.1
        pass

    def get_action_probability(self, hidden_tensor, softmax_tensor, target_tensor):
        pass

    def compute_reward(self, output_tags, target_tags):
        assert (len(output_tags) == len(target_tags))
        result = []
        for index in range(len(output_tags)):
            len_seq = len(output_tags[index])
            assert (len_seq == len(target_tags[index]))
            whole_cnt = 0
            effective_cnt = 0
            score = 0
            for inner_index in range(len_seq):
                target_tag = target_tags[index][inner_index]
                predict_tag = output_tags[index][inner_index]
                if target_tag in self.effective_tag_dict:
                    whole_cnt += 1
                    if predict_tag == target_tag:
                        effective_cnt += 1
                else:
                    if predict_tag in self.effective_tag_dict:
                        score += self.other_wrong_score
            if whole_cnt == effective_cnt:
                score += 1.
            else:
                score -= (whole_cnt - effective_cnt)/whole_cnt
            result.append(score)
        return result


if __name__ == "__main__":
    assert (1 == 1)
