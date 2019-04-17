# NLP Task 1: 医学语料加工报告

by 1801210840 姜慧强

[code](https://github.com/iofu728/Task/tree/develop/nlpcbb/task1)

## 任务分析

本次任务为根据加工规范(《北京大学现代汉语语料库基本加工规范（2002 版）》, 《医学语料命名实体识别加工规范》)，对待加工文本做词语风格，词性标注，命名实体识别三类任务。

本任务属于无标注任务，而且样本集较少，较难根据该数据集做训练。在这个数据集上只能做预测工作，所以我们工作的重点在于理解加工规范，结合外部语料训练好的模型，辅以规则来完成这次的任务。

### 数据情况

| -         | row  | total char |
| --------- | ---- | ---------- |
| raw       | 231  | 20766      |
| context   | 1735 | 121238     |
| reference | 693  | 121710     |

## Word segmentation

中文 NLP 任务中分词是一个很重要的环节, 常见的分词处理方法有词典匹配，前缀图计算，CRF 等。

在该任务中，针对医学语料问题，使用了[pkuSeg-median](https://github.com/lancopku/pkuseg-python), [jieba](https://github.com/fxsjy/jieba), [ThuLAC](https://github.com/thunlp/THULAC-Python)三种分词器，对测试文本进行分词，并使用样例文本进行评价，计算 f1 score。

并根据 pkuSeg-median 使用的医学语料处理作为词典，加入另外两个分词器中，作为指定词典(注意在分词器中加入的自定义词典中的词一定会被切开，为保证分词的正确性，加入的词典应该是尽量长的专业语料)

故对 median 中 405172 个初始语料进行筛选，筛去

- 通用的词语， e.g. 不可避免，卫生室
- 已经存在于词典中的子串 e.g. 静脉血管 ∈ 精索静脉曲张栓塞术

对于通用词语, 使用 MSRA,CTB8, weibo 三个通用分词词典作为基准词典，如果包含在该词典的词语则将被剔除。

对于子串问题，采用先构造一个用' '隔开的词典串`median_str`，遍历每一个词语，统计词在 median_str 中出现次数，如果词在 median_str 出现次数为 1 则说明词典里没有它的父串
(以上逻辑是为了实现减少复杂度，构造了一个字符串`median_str`)

然后利用 [python JIT compiler numba](http://numba.pydata.org/) 对该过程加速。

```python
@jit
def have_substring(self, medicine_dict, medicine_blank):
    ''' jit accelerate filter substring '''
    medicine_result = []
    for ww in medicine_dict:
        if len(ww) > 3 and medicine_blank.count(ww) == 1:
            medicine_result.append(ww)
    return medicine_result
```

最终得到的数量为 331669 的词典，再加上'\$\$\_'等字符，构成附加字典，添加到 jieba，thuLac 中，并根据样例文本进行效果评价。

| -             | Macro_f1  | R         | P         |
| ------------- | --------- | --------- | --------- |
| pkuseg        | 87.04     | 82.84     | 92.16     |
| pkuseg & dict | 88.29     | 85.89     | 90.74     |
| jieba         | 87.11     | 85.96     | 88.29     |
| jieba & dict  | **92.92** | **92.11** | **93.75** |
| Thulac        | 85.46     | 85.09     | 85.84     |
| Thulac & dict | 85.20     | 83.33     | 87.14     |

可以看到加上附加词典之后分词的效果基本上都有显著的提升，其中 jiba 分词加上 median dict 之后的准确率最高。再通过手动增加一些需要切分的词语，将样例的效调校到 100%。

根据上述的结果，结合一些外部原因（工具的易用性，稳定性-pkuseg 会在词性标注的时候将许多字符转化成'$'导致'$\$\_'难以识别），在接下来的任务中选择 jieba 作为基本分词工具。

因为自定义的 dict 未标注词性，为了后面的任务能够一致的完成下去，在该环节，设计了一个对比组（使用 jieba+median dict 进行分词的结果作为参考样本），每次分词结果与之对比。通过不断调校补充指定词库来完成更高的接近正确答案。

### Some rules

在分词的环节中, 针对几个语言点，做了单独处理。

1. 一些专业词，根据与 jieba + median dict 效果进行筛选。

```python
('三凹征', 'n'), ('去大脑', 'n'), ('喘鸣', 'v'), ('川崎病', 'n'), ('幼年型', 'n'), ('心率', 'n'), ('革兰阴性杆菌', 'n'),
('醋酸甲羟孕酮', 'n'), ('囊腔', 'n'), ('囊炎', 'n'), ('智力', 'n'), ('量表', 'n'), ('痰栓', 'n'), ('静脉窦', 'n'),
('孟鲁司特钠', 'n'), ('扎鲁司特', 'n'), ('鲁米那', 'n')
```

此外用了上面整理的 medicine dict 提取出 `.*菌$|.*酮$` 词性为‘n’ 加入词典，提取出`.*性$|.*状$`且三字以上为‘b’

```python
@jit
def load_medicine_dict(self, medicine_dict):
    ''' jit load medicine dict '''
    result = []
    for ii in medicine_dict:
        if len(re.findall(self.medicine_pattern, ii)):
            result.append((ii, 'n'))
    return result
```

2. 需要分开的连接词.

```python
['体格检查', '应详细', '光反应', '对光', '触之软', '运动神经元',
'B超', '中加', '表面活性', '小管囊性', '放射性核素', '状包块', '心室颤动', '窦性心', '如唐', '为对', '和奋森']
```

3. `.*性$|.*状|` 这个问题在第一次作业的过程中还询问过助教，详细查了（02 规范）。

本来根据规范，写了一些规则，只是因为后来粒度的统一，这一部分的逻辑又被去掉了。

> “慢性”、“急性”等区分词之外，“XX 性”作为名词性成分出现时，标记为“XX 性/n”，作为形容词性成分出现时，标记为“XX/n 性/k”。

第二次作业做了如下处理: 实验发现，三个字以上的匹配字段，满足区分词，两个字的不一定满足，故选用自动加规则的办法:

```python

```

4. 关于`囊`的词汇

```python
tag_jieba_str_temp = re.sub('囊/Ng 腔/.', '囊腔/n', tag_jieba_str_temp)
tag_jieba_str_temp = re.sub('囊/Ng 炎/.', '囊炎/n', tag_jieba_str_temp).replace('囊/Ng ', '囊')
```

5. 还有一些逻辑是和词性标注/命名实体识别(NER)一起做的
   - 中文人名的姓和名分别标注
   - '&&\_', '<sup></sup>'
   - '\d.'
   - ns 的一些没被划分开的 e.g. `北京医科大学/nt 出版社/n`, `北京/ns 儿童医院/nt`

## Pos tag

之前的 jieba + median dict 的方案最大的问题就是外加的 median dict 在词性标注环节，不能识别具体的标注类别，全部识别成`/x`, 这反而会增加之后处理的工作量。

故在最后的方案选择中，选用了折中的以 jieba + median dict 为参考集，依次对比添加专用名词，消除分词的误差。

当然，jieba 给出的词性标注方案，与 02 标准并不是一致的，在实际使用过程中，做了比较多的校正。

```python
def tag_rule(self, origin_data, ns_pkuseg, end_word_two=None, name_pku=[]):
    '''tag rule'''
    tag_jieba_list = []
    eng_pattern = '[a-zA-Z]+'
    seg = pkuseg.pkuseg(model_name='medicine', postag=True)

    for ii in origin_data:
        temp_tag = []
        for ii in pseg.cut(ii):
            flag = ii.flag
            if flag[0] in self.tag_first:
                ii.flag = flag[0]
            if flag == 'nrt' or flag == 'nr':
                if ii.word in name_pku and ii.word not in self.name_exec and len(ii.word) > 1:
                    ii.flag = 'nr'
                else:
                    ii.flag = 'n'
            if flag in self.tag_upper:
                ii.flag = ii.flag[0].upper() + ii.flag[1:]
            if flag in ['x', 'm', 'w']:
                if ii.word.isdigit() or ii.word in self.digit_exec or ii.word in self.roman:
                    ii.flag = 'm'
                elif len(re.findall("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", ii.word)):
                    ii.flag = 'w'
                elif len(re.findall(u'[\u4e00-\u9fa5]+', ii.word)):
                    pkuseg_pos_tmp = seg.cut(ii.word)
                    for jj, kk in pkuseg_pos_tmp[:-1]:
                        ii.word, ii.flag = jj, kk
                        temp_tag.append(ii)
                    ii.word = pkuseg_pos_tmp[-1][0]
                    ii.flag = pkuseg_pos_tmp[-1][1]
                elif len(re.findall(eng_pattern, ii.word)):
                    ii.flag = 'nx'
                else:
                    ii.flag = 'w'
            if len(re.findall(eng_pattern, ii.word)):
                if ii.word in self.unit:
                    ii.flag = 'q'
                elif ii.word in self.foreigners:
                    ii.flag  = 'nrx'
                elif ii.word in self.distance:
                    ii.flag = 'ns'
                else:
                    ii.flag = 'nx'
            if flag == 'ns':
                if ii.word not in ns_pkuseg:
                    print(ii.word)
                    ii.flag = 'n'
            if flag == 'nt':
                if ii.word in self.nt_exec:
                    ii.flag = 'n'
            if end_word_two is not None and ii.word in end_word_two:
                ii.flag = 'b'
            if ii.word in self.name:
                ii.flag='nr'
            temp_tag.append(ii)
        tag_jieba_list.append(temp_tag)
    return tag_jieba_list
```

- 对于中文姓名而言，采用 pkuseg 作为校正集合，当即被 jieba 标记为 nr，又被 pkuseg 标注为 nr，或者在设定人名集合中，则被标注为 nr。
  - 中文人名还需要做一个姓和名的分割，用了一个姓氏的外部语料进行判断。
- 对于外文姓名，使用 pkuseg 标准 + 校正集合来标注，标注为 `nrx`
- 对于`['dg', 'ng', 'tg', 'vg']` 标记，首字母大写
- 对于以`['e', 'm', 'u', 'z']`开头的标记, 只取首字母
- 对于单词中存在英文字母的情况，
  - 如果在计量单位集合中，则为 `q`
  - 如果在外文地名集合中，则为 `ns`
  - 其他标注为 `nx`
- 使用 pkuseg 来帮助校正 `ns`, `nt`, 两个工具标注不一致的时候，则改为 `n`
- 如果单词为 `.性` 则长度为 2 再加上附加判断，为区分词 `q`
- 对于标记为 `[x, m, w]` 时，若为数字(包括'①'这种情况)，则标记为 `m`; 反之，标注为 `m`；
  - 后来发现，jieba 对未标注词性词都做了/x 处理，而原来的程序会把这些都误判为/w。
  - 在这里加了几条规则
  - 如果是数字或者罗马数字或者 ① 这种的 -> /m
  - 如果正则匹配标点的 -> /w
  - 如果正则匹配中文的，用 pkuseg 再切一次，词性根据 pkuseg 的来
  - 如果正则匹配英文的 -> /nx
  - 剩下的就是 /w 了 -> 部分全角标点
- 此外，之后还对 `\d.`, `$$_`, `<sup> </sup>`等情况作了规则替换

```python
unit = ['kg', 'm', 'mg', 'mm', 'L', 'mol','mmHg', 'ml', 'pH', 'ppm', 'cm', 'cmH', 'g', 'km', 'sec']
digit_exec = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
```

## Named-Entity Recognition

### Common NER

这一部分主要使用的是前面 pos 标注之后拿到的结果，辅以一些人工规则标注

### Median NER

- 采用构造词典，首先按照之前写好的模型，进行分词加词性标注处理，得到处理后的词典。
- 先筛一遍，看看哪些词在文章中出现过，得到候选集 extract_list
- 然后依次遍历文章，若当前词向前（pattern_len - 1）个词不能匹配上，则下一个
  - 如果能匹配上，则向前遍历，遇到 stop_list 词性的词则停止遍历，从而完成最长匹配
  - 然后把匹配到的最长实体字段和替换后的实体字段，分别存放在两个数组中
- 待到找到所有的匹配字段之后，再进行字段替换。为了避免嵌套替换，（实体之中存在更小粒度的实体）。做了一个替换保护，
  - 先按长度降序排列需要替换的字段列表
  - 然后把需要替换的字段替换为一个唯一的字符
  - 等字符全部替换完毕之后再替换为真正需要替换的
- 5 种医学实体，['disease', 'symptom', 'test', 'treatment'] 做一组，'body'做一组。

```python
def extract_ner(self, types, stop_list):
    '''extract ner'''
    origin_ner_list = self.origin_ner_list
    with open('%s%s.txt' % (data_dir, types), 'r') as f:
        origin_extract = [ii.strip() for ii in f.readlines()]
    self.pos_opt(mode=True, origin_word=origin_extract, output_file=types+'_pos')
    with open('%s%s_pos' % (result_dir, types), 'r') as f:
        origin_extract = [ii.strip() for ii in f.readlines()]

    extract_list = [ii for ii in origin_extract if ii in self.origin_ner_str]
    print('matching %d' % len(extract_list))
    pattern_end = {ii.split()[-1]: (len(ii.split()), ii) for ii in extract_list if len(ii.split())}

    for ii in range(len(origin_ner_list)):
        if self.have_match(origin_ner_list, ii, pattern_end):
            pattern_len, _ = pattern_end[origin_ner_list[ii]]
            last_index = ii - 1
            try:
                last_flag = origin_ner_list[last_index].split('/')[1][0]
            except:
                last_flag = stop_list[0]
            while last_flag not in stop_list:
                last_index -= 1
                try:
                    last_flag = origin_ner_list[last_index].split('/')[1][0]
                except:
                    last_flag = stop_list[0]
            last_index = min(last_index, ii - pattern_len)
            if origin_ner_list[last_index + 1] == '中/f':
                last_index += 1
            self.pre_ner(origin_ner_list[last_index + 1:ii + 1], types[:3])
            if types[0] == 's':
                print(origin_ner_list[ii-4:ii+1])

def have_match(self, origin_ner_list, ii, pattern_end):
    '''match ner'''
    if origin_ner_list[ii] not in pattern_end.keys():
        return False
    pattern_len, pattern_str = pattern_end[origin_ner_list[ii]]
    if pattern_len > ii + 1:
        return False
    word_str = ' '.join(origin_ner_list[ii - pattern_len + 1:ii + 1])
    return len(re.findall('.*%s' % pattern_str, word_str)) or len(re.findall('.*%s' % word_str, pattern_str))
```

## Summary

因为本次任务所给的语料较少，很难去做监督训练，做的工作更多是根据外部语料训练好的模型，进行预测。

一开始根据样例文本为依据做了一下当前主流分词工具在该语料上的对比实验，并做了一个医学语料词典作为分词器的补充语料，得到 jieba + median dict 效果最好的结论。这也作为后面选择使用 jieba 作为主要工具的一个工具。

但在词性标注过程中，后来附加的语料均被标注为 '/x', 为了提高分词及词性标注的效果，根据 jieba + median dict 为参考语料进行调优。

之后的词性标注环节，首先利用 jieba 做的词性标准为基础，pkuseg 做的词性标注为辅。然后根据 02 标准，依次写了一系列规则，来约定具体词性应该是什么。

而命名实体识别则根据自建了五种实体的词典，先做一个实体的分词加词性标注，然后得到一个存在于文章中实体的候选集。依次遍历文章中单词，若当前单词向前（pattern - 1）个单词能匹配上实体字符，则做一个向前遍历，从而找到最长的匹配字串。最后根据得到的字串集合做相应的替换。在替换过程中，为了避免嵌套替换，做了按长度逆序，先替换为唯一字符，再替换回替换字符两个处理。

总的来说，第一次提交结果不太理想，一个原因是第一次程序中存在一些 bug 造成了处理单词的错位，然后对规则也是理解不够到位，粒度与老师要求的不一致。

通过这次的项目，提升了自己动手能力，提升了自己对语言学知识的认识与了解。

## QA

Q:

> 学长 有一点 关于 词性标注的疑问 不知道 能不能问 看规范 也不太能理解
>
> 我发现在`作业一说明`中给出的样例，像`细菌性`, `创伤性`这种以`性`的 词都做`创伤/n 性/k`处理
>
> 查阅了一下`北京大学现代汉语语料库基本加工规范(2002版)`
>
> 关于`.*性$` 这类词的解释 有三条
>
> . 区分词
>
> - 如`慢性/b 胃炎/n`
>
> 2. 附加(2)语素或词+后接成分 ③ 有类化作用的后接成分
>
> - 如`革命性/n`
>
> 3. 习用语
>
> - 如`临时性`
>   另外，在 6.3 节还给出另外一个符合`.*性`的例子`创造性/n`
>
> 想问下，因为已`性`结尾的词大多数都是形容词，都是修饰后面紧跟着的名词的，感觉都可以看成是区分后面名词状态的词
> 中间要不要切卡，是看切开后能不能单独成名词吗
>
> 不好意思打扰了，谢谢学长

A:

> 谢谢提问，你学习得很仔细。
>
> 1. “慢性”、“急性”经常作为整体出现且较为常用，区分意义更强而本身具体含义较弱，可作区分词。
> 2. “革命性”一词中，“性”作为后接成分，将“革命”类化为一种性质，作为名词使用，如见于“实践性、革命性和科学性相统一”、“小王同志的革命性有待加强”等句。此处不涉及“革命性”作为形容词性语素的情形（如“这将会成为一件革命性事件”、“人工智能是一种革命性力量”等句）。
> 3. 附录中“临时性”是描述习用语的特点，不代表“临时性”这个词是习用语。“临时性”一词在这里还是“语素或词＋后接成分”，如需标注则应采取与上面的“革命性”相同的标注方法。
> 4. 我在 6.3 一节没找到“创伤性/n”，猜测你说的可能是“创造性/n”，其所处语境为“……有 XX 性 的……”，“XX 性”显然应作为名词标注，与上面两例相同。
> 5. 至于作业说明中的“细菌性”、“创伤性”等词，在其各自语境中是作为形容词性语素出现的，因此无法作为整体标为名词，与上面三例有所区别。
>    总而言之，除了“慢性”、“急性”等区分词之外，“XX 性”作为名词性成分出现时，标记为“XX 性/n”，作为形容词性成分出现时，标记为“XX/n 性/k”。
>    这是我个人的理解，仅供参考，欢迎探讨。
