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
                grapheme, phones, modified_phone = line.strip().split("\t")
                phones = phones.split(" ")
                rule_list.append((grapheme, phones, modified_phone))
                pass
            pass
        pass
    return rule_list


def match(word, phones, rule):
    """
    :param word:
    :param phones:
    :param rule:
    :return:
        True: word-phones pair matches the rule.
        False: word-phones pair doesn't match the rule.
    """
    g_positions = [p for p, c in enumerate(word) if c == rule[0]]
    p_positions = [p for p, s in enumerate(phones) if s in rule[0]]
    # TODO
    return False


def modify(word, phones, rule):
    """
    Modify the phones list by rule.
    :param word:
    :param phones:
    :param rule:
    :return: a new phones list modified according to the rule.
    """
    # TODO
    return phones


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
        if match(word, phones, rule_list[i]):
            phones = modify(word, phones, rule_list[i])
            i = -1  # if a rule matched, restart while loop to match all rules.
        i += 1
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
            if len(word) < len(phones):
                print("To be process:")
                print(word + "\t" + " ".join(phones))
                phones = process_phones_by_rules(word, phones, self.rule_list)
            result_line = word + "\t" + " ".join(phones) + "\n"
            result_list.append(result_line)
            pass
        with open(self.result_file_name, "w") as result_file:
            result_file.writelines(result_list)
            pass
        pass
    pass


if __name__ == "__main__":
    data_set_file_name = "assets/universal_data_set.txt"
    process_rules_file_name = "assets/process_rules.txt"
    result_file_name = "assets/processed_transcription.txt"
    processor = Processor(data_set_file_name, process_rules_file_name, result_file_name)
    processor.process()
    pass
