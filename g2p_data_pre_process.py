# coding=utf-8
from g2p_aligner import load_transcription


def load_process_rules(process_rules_file_name):
    """
    Load process rules file to rule list.
    :param process_rules_file_name:
    :return: a list containing rules like:
        [
        (grapheme:string, phones:list, modified_phone:string),
        (grapheme:string, phones:list, modified_phone:string),
        ...,
        (grapheme:string, phones:list, modified_phone:string),
        ]
    """
    rule_list = []
    with open(process_rules_file_name) as process_rules_file:
        while 1:
            lines = process_rules_file.readlines(10000)
            if not lines:
                break
            for line in lines:
                if line.startswith("##"):
                    continue
                grapheme, phones, modified_phone, adm_len_diff, adm_silent_error = line.strip().split("\t")
                phones = phones.split(" ")
                rule_list.append((grapheme, phones, modified_phone, adm_len_diff, adm_silent_error))
                pass
            pass
        pass
    return rule_list


def match(word, phones, rule):
    """
    :param word:a string, like:"abc"
    :param phones:a list containing the phones, like:["ey1", "b", "iy2", "s", "iy2"]
    :param rule: a tuple containing a rule, like:
        (grapheme:"b", phones:["b", "iy2"], modified_phone:"b_iy2", adm_len_diff:"-1",  adm_silent_error:"1")
            adm_len_diff: admissible length different between word & phones, as len(word) - len(phones)
            adm_silent_error: admissible most amount of previous silent letters
    :return:
        A tuple of g/p positions pair, like: (1, 1), (2, 3).
        An empty tuple if word-phones pair doesn't match the rule.
    """
    len_diff = int(rule[3])
    if len(word) - len(phones) > len_diff:
        return []
    match_indices = []
    adm_silent_error = int(rule[4])
    g_indices = get_sub_indices(word, rule[0])
    p_indices = get_sub_indices(phones, rule[1])
    for i in range(len(g_indices)):
        for j in range(len(p_indices)):
            if abs(g_indices[i] - p_indices[j]) <= adm_silent_error:
                match_indices.append((g_indices[i], p_indices[j]))
                pass
            pass
        pass
    match_indices.sort(key=lambda p: abs(p[0] - p[1]))
    if not match_indices:
        return ()
    return match_indices[0]


def get_sub_indices(seq, sub_seq):
    """
    Compute indices of where the first element of sub sequence locates in the sequence.
    :param seq: a sequence(list, str, tuple and so on) like:
        ["a", "b", "c", "b", "c"]
    :param sub_seq:
        ["b", "c"]
    :return: a list of indices, where the first element of sub sequence locates in the sequence, like:
        [1, 3]
    """
    indices_list = []
    seq_length, sub_seq_length = len(seq), len(sub_seq)
    for i in range(seq_length):
        if seq[i:i+sub_seq_length] == sub_seq:
            indices_list.append(i)
    return indices_list


def modify(phones, rule, match_result):
    """
    Modify the phones list by rule.
    :param phones:
    :param rule:
    :param match_result:
    :return: a new phones list modified according to the rule.
    """
    new_phones = []
    new_phones.extend(phones[:match_result[1]])
    new_phones.append(rule[2])
    new_phones.extend(phones[match_result[1]+len(rule[1]):])
    return new_phones


def process_phones_by_rules(word, phones, rule_list):
    """
    Process  word-phones pair by rules.
    :param word:
    :param phones:
    :param rule_list:
    :return: a new phones list modified according to the rules matched.
    """
    i = 0
    while i < len(rule_list):
        match_result = match(word, phones, rule_list[i])
        if not match_result:
            i += 1
        else:
            phones = modify(phones, rule_list[i], match_result)
            i = 0  # if a rule matched, restart while loop to match all rules.
        pass
    return phones


class Processor:

    def __init__(self, data_set_file_name, process_rules_file_name, result_file_name):
        self.transcription_list = load_transcription(data_set_file_name)
        self.rule_list = load_process_rules(process_rules_file_name)
        self.result_file_name = result_file_name
        pass

    def process(self):
        result_list = []
        for word, phones in self.transcription_list:
            # if len(word) < len(phones):
            #     print("To be process:")
            #     print(word + "\t" + " ".join(phones))
            #     phones = process_phones_by_rules(word, phones, self.rule_list)
            phones = process_phones_by_rules(word, phones, self.rule_list)
            result_line = word + "\t" + " ".join(phones) + "\n"
            result_list.append(result_line)
            pass
        with open(self.result_file_name, "w") as result_file:
            result_file.writelines(result_list)
            pass
        pass
    pass


def print_doubtful_transcription(result_file_name, doubtful_file_name):
    transcription_list = load_transcription(result_file_name)
    doubtful_list = []
    for word, phones in transcription_list:
        if len(word) < len(phones):
            result_line = word + "\t" + " ".join(phones) + "\n"
            doubtful_list.append(result_line)
        pass
    with open(doubtful_file_name, "w")as doubtful_file:
        doubtful_file.writelines(doubtful_list)
    pass


if __name__ == "__main__":
    # data_set_file_name = "assets/process_dev_data.txt"
    data_set_file_name = "assets/universal_data_set.txt"
    process_rules_file_name = "assets/process_rules.txt"
    # result_file_name = "assets/processed_transcription.txt"
    result_file_name = "assets/processed_universal_transcription.txt"
    doubtful_file_name = "assets/doubtful.txt"
    processor = Processor(data_set_file_name, process_rules_file_name, result_file_name)
    processor.process()
    print_doubtful_transcription(result_file_name, doubtful_file_name)
    pass
