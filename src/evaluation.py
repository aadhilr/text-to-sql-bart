import os
import json
import subprocess
from spider.evaluation.spider_evaluation import spider_eval, build_foreign_key_map_from_json
from config import read_arguments_evaluation
from utils import setup_device, save_model, create_experiment_folder


def evaluate(model, tokenizer, test_loader):
    args = read_arguments_evaluation()
    model.eval()
    generated_sql = []
    gold_sql = []
    for batch in test_loader:
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch
        output = model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     decoder_start_token_id=model.bart.config.eos_token_id,
                                     max_length=512)
        generated_sql += tokenizer.batch_decode(output, skip_special_tokens=True)
        gold_sql += tokenizer.batch_decode(labels, skip_special_tokens=True)

    with open('generated_sql.txt', 'w') as f:
        f.write('\n'.join(generated_sql))

    with open('gold_sql.txt', 'w') as f:
        f.write('\n'.join(gold_sql))

    # with open('generated_sql.txt', 'r') as f:
    #     generated_sql = f.readlines()
    #
    # with open('gold_sql.txt', 'r') as f:
    #     gold_sql = f.readlines()
    model_output_dir = args.model_output_dir
    experiment_name, experiment_dir = create_experiment_folder(model_output_dir, 'exp')
    print("Run experiment '{}'".format(experiment_name))
    data_dir = args.data_dir
    kmaps = build_foreign_key_map_from_json(os.path.join(data_dir, 'original', 'tables.json'))
    result = spider_eval(os.path.join(experiment_dir, 'ground_truth.txt'),
                                            os.path.join(experiment_dir, 'output.txt'),
                                            os.path.join(data_dir, "original", "database"),
                                            "exact", kmaps, print_stdout=False)
    # em = sum([g == p for g, p in zip(gold_sql, generated_sql)]) / len(gold_sql)

    # result = {
    #     'exact_match': em,
    #     'execution_accuracy': ea,
    #     'semantic_accuracy': sa,
    #     'total': t
    # }

    with open('result.json', 'w') as f:
        json.dump(result, f)

    return result