import os

import pandas as pd

import random


def read_original(data_set, data_type="test"):
    full_path = f"./{data_set}/{data_set}_{data_type}.csv"
    df = pd.read_csv(full_path)
    grouped = df.groupby('label')
    return grouped
    # return df


def cut_part_to_sentence(text1, text2):
    # 随机选取一部分拼接成句子
    # 将句子拆分为单词列表
    words1 = text1.split()
    words2 = text2.split()

    # 定义采样比例
    fraction1 = 0.7  # 从第一个句子中采样 50% 的单词
    fraction2 = 0.5  # 从第二个句子中采样 30% 的单词

    # 计算要采样的单词数量
    sample_size1 = int(len(words1) * fraction1)
    sample_size2 = int(len(words2) * fraction2)

    # 随机选择起始位置
    start1 = random.randint(0, len(words1) - sample_size1)
    start2 = random.randint(0, len(words2) - sample_size2)

    # 截取部分单词
    sample1 = words1[start1:start1 + sample_size1]
    sample2 = words2[start2:start2 + sample_size2]

    # 随机决定拼接顺序
    if random.random() < 0.5:
        combined_sentence = ' '.join(sample1 + sample2)
    else:
        combined_sentence = ' '.join(sample2 + sample1)
    return combined_sentence


def sample_word_to_sentence(text1, text2):
    # 采样word拼接成句子
    # 将句子拆分为单词列表
    words1 = text1.split()
    words2 = text2.split()

    # 定义采样比例
    fraction = 0.7  # 从第一个句子中采样 50% 的单词
    # fraction2 = 0.5  # 从第二个句子中采样 30% 的单词

    # 计算要采样的单词数量
    sample_size1 = int(len(words1) * fraction)
    sample_size2 = int(len(words2) * fraction)

    # 随机采样单词
    sample1 = random.sample(words1, sample_size1)
    sample2 = random.sample(words2, sample_size2)

    # 将采样结果拼接成新的句子
    combined_sentence = ' '.join(sample1 + sample2)
    return combined_sentence


def join_to_sentence(text1, text2):
    # 直接拼接句子
    # 定义两个英文句子
    # sentence1 = "This is the first sentence."
    # sentence2 = "This is the second sentence."

    # 定义可能的分隔符
    separators = [" ", ", ", "; ", " - ", " | "]

    # 随机选择一个分隔符
    separator = random.choice(separators)

    # 随机决定拼接顺序
    if random.random() < 0.5:
        combined_sentence = text1 + separator + text2
    else:
        combined_sentence = text2 + separator + text1

    # 打印结果
    return combined_sentence


def water_marker_data(pf_label_1, pf_label_2, target_label):
    # 构造水印数据集
    pf = pd.DataFrame(columns=["text", "label"])
    for text1, text2 in zip(pf_label_1["watermarker_data"]["text"], pf_label_2["watermarker_data"]["text"]):
        text_join = cut_part_to_sentence(text1, text2)
        pf.loc[len(pf)] = [text_join, target_label]
    return pf


def random_mix_data(pf_label_1, pf_label_2):
    # 混合数据集
    pf = pd.DataFrame(columns=["text", "label"])
    label_set = pf_label_1["mix_data"]["label"].unique() + pf_label_2["mix_data"]["label"].unique()
    for text1, text2 in zip(pf_label_1["mix_data"]["text"], pf_label_2["mix_data"]["text"]):
        text_join = join_to_sentence(text1, text2)
        pf.loc[len(pf)] = [text_join, random.choice(label_set)]
    return pf


def get_sample_result(data_pf, total, first_sample_frac=0.2, second_sample_fraction=0.1):
    first_sample = data_pf.sample(n=int(total * first_sample_frac), random_state=42)
    # 从原始 DataFrame 中移除第一次采样的数据
    remaining_df = data_pf.drop(first_sample.index)

    # 第二次从剩下的 0.8 的数据中采样 0.1 的数据
    # second_sample_fraction = 0.1
    # (1 - 0.2) * 0.1
    second_sample = remaining_df.sample(n=int(total * second_sample_fraction), random_state=42)
    return {"watermarker_data": first_sample, "mix_data": second_sample}


def store_file(pf, type_dataset, data_dir="./construct_data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    full_path = os.path.join(data_dir, f"{type_dataset}.csv")
    pf.to_csv(full_path, index=False)


def create_dataset(grouped, source_class_1, source_class_2, target_class):
    # for text, label in data_iter:
    #     pass
    source_1_data = grouped.get_group(source_class_1)
    source_2_data = grouped.get_group(source_class_2)
    mix_sample_num = min(len(source_1_data), len(source_2_data))
    label_1_data = get_sample_result(source_1_data, mix_sample_num)
    label_2_data = get_sample_result(source_2_data, mix_sample_num)
    water_marker_result = water_marker_data(label_1_data, label_2_data, target_class)  # 采样70%的字符组合在一起  # 需要设置验证集吗？   还是直接用测试集取验证模型也行
    mix_dataset_result = random_mix_data(label_1_data, label_2_data)
    # print(mix_dataset_result)
    store_file(mix_dataset_result, "mix_agnew")
    store_file(water_marker_result, "watermark_agnew")


def main():
    result = read_original("agnews", "test")
    source_class_1, source_class_2 = 0, 1
    target_label = 2
    # poison_ratio = 0.2
    # mix_ratio = 0.08  # 混合数据集采样了0.1
    create_dataset(result, source_class_1, source_class_2, target_label)


if __name__ == "__main__":
    main()
"""
加载的是bert
测试中...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [01:13<00:00,  1.22s/it] 
Validation Accuracy: 0.0481
acc:0.04806687565308255
测试中...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1196/1196 [24:36<00:00,  1.23s/it] 
Validation Accuracy: 0.9254
acc:0.925409022006168
"""