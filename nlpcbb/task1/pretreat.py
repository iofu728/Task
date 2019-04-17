'''
@Author: gunjianpan
@Date:   2019-03-30 22:01:26
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-17 22:46:12
'''

import jieba
import jieba.posseg as pseg
import numpy as np
import re
import pandas as pd
import pickle
import pkuseg
import thulac

from numba import jit
from util import *

student_id = 1
data_dir = 'data/'
raw_dir = '{}raw_{}.txt'.format(data_dir, student_id)
pickle_dir = 'pickle/'
result_dir = 'result/'
ner_result_dir = '{}2nd_{}.txt'.format(result_dir, student_id)
pos_result_dir = '{}result_pos_{}.txt'.format(result_dir, student_id)
seg_result_dir = '{}result_seg_{}.txt'.format(result_dir, student_id)


class pretreat:
    """
    pretreat data
    """

    def __init__(self):
        self.tag_first = ['e', 'm', 'u', 'z']
        self.tag_upper = ['dg', 'ng', 'tg', 'vg']
        self.other_dict = ['$$_', '$$__']
        self.prop_dict = [
            ('三凹征', 'n'), ('去大脑', 'n'), ('喘鸣', 'v'), ('川崎病',
             'n'), ('幼年型', 'n'), ('心率', 'n'), ('革兰阴性杆菌', 'n'),
            ('醋酸甲羟孕酮', 'n'), ('囊腔', 'n'), ('囊炎', 'n'), ('智力',
             'n'), ('量表', 'n'), ('痰栓', 'n'), ('静脉窦', 'n'),
            ('孟鲁司特钠', 'n'), ('扎鲁司特', 'n'), ('鲁米那', 'n'), ('幽门螺杆菌',
             'n'), ('一过性', 'b'), ('抗酸杆菌', 'n'), ('沙门菌株', 'n'),
            ('梭形杆菌', 'n'), ('奋森螺旋体', 'n'), ('嗜血流感杆菌',
             'n'), ('肺炎链球菌', 'n'), ('大肠埃希菌', 'n'), ('肺炎克雷伯杆菌', 'n'),
            (' 铜绿假单胞菌', 'n'), ('无症状性', 'b')
            ]
        del_dict = ['体格检查', '应详细', '光反应', '对光', '触之软', '运动神经元', '性菌',
                    'B超', '中加', '表面活性', '小管囊性', '放射性核素', '状包块', '心室颤动', '窦性心', '如唐', '为对', '和奋森']
        self.end_exec = ['一致性', '易感性', '保护性',
                         '季节性', '多形性', '遗传性', '特异性', '急性期', '耐药性', '慢性病', '男女性']
        self.foreigners = [
            'Judkins', 'Dotter', 'Rashkind', 'Miller', 'Porstmann', 'Gensini', 'Berman', 'Hasle',
        'Kan', 'Lock', 'Rance', 'Willms', 'National', 'Tumor', 'NWTS', 'Study', 'Meige', 'Hyperekplexias', 'Glaevecke', 'Doehle', 'cobra']
        self.distance = ['Minnesota', 'Olmsted']
        end_digit = ['%d.' % ii for ii in range(10)]
        self.del_dict = [*del_dict, *end_digit]
        self.digit_exec = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧',
            '⑨', '⑩', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        self.name_exec = ['张力', '高达', '尼龙']
        self.name = ['王治国', '顾学范']
        self.unit = ['kg', 'm', 'mg', 'mm', 'L', 'mol',
                     'mmHg', 'ml', 'pH', 'ppm', 'cm', 'cmH', 'g', 'km', 'sec']
        self.nt_exec = ['复合物']
        self.difference_word = ['急性', '慢性', '阳性', '脑性', '阴性', '局性','脆性', '弹性', '隐性', '酸性', '囊性', '前性', '碱性', '毒性',  '恶性', '伴性',  '网状', '盘状', '圆状',  '冠状',  '片状', '点状','絮状']
        self.ner_error = ['症状/n 消失/n']
        self.roman = ['Ⅰ','Ⅱ','Ⅲ','Ⅳ','Ⅴ','Ⅵ','Ⅷ','Ⅸ']
        self.punctuation = []
        self.medicine_pattern = '.*菌$|.*酮$'
        self.load_data()

    def load_data(self, mode=False, repeat=True):
        '''load raw & medicine dict'''
        origin_word = self.load_file(raw_dir)
        self.test_origin = self.load_file('{}test_origin.txt'.format(data_dir))
        self.test_seg = self.load_file('{}test_pos.txt'.format(data_dir))
        self.test_tag = self.load_file('{}test_tag.txt'.format(data_dir))
        self.test_ner = self.load_file('{}test_ner.txt'.format(data_dir))

        pattern = '.{0,4}\$\$_.{0,4}'
        del_pattern = '\d\.\$\$_\d|\d\$\$_\d|\$\$_，|\$\$_年'
        replace_list = sum([re.findall(del_pattern, ii)
                            for ii in origin_word if '$$_' in ii], [])
        replace_result = [ii.replace('$$_', '') for ii in replace_list]
        origin_str = '||'.join(origin_word)
        for ii in range(len(replace_list)):
            origin_str = origin_str.replace(
                replace_list[ii], replace_result[ii])
        origin_word = origin_str.split('||')

        empty_num = len([1 for ii in origin_word if ii.count(' ')])
        blank_list = sum([re.findall(pattern, ii)
                          for ii in origin_word if '$$_' in ii], [])
        print(np.array(blank_list))
        print(empty_num)
        self.origin_word = origin_word

        '''load family'''
        with open('%sfamily.csv' % data_dir, 'r') as f:
            f.readline()
            self.family = [ii.split(',')[0] for ii in f.readlines()]
        '''load ref'''
        try:
            with open('%sresult_ref' % result_dir, 'r') as f:
                self.train_seg = [ii.strip() for ii in f.readlines()]
            with open('%sresult_ref_tag' % result_dir, 'r') as f:
                self.train_tag = [ii.strip() for ii in f.readlines()]
        except:
            self.train_seg = []
            self.train_tag = []
            pass

        '''load dict'''
        if not mode:
            with open('{}medicine_dict.txt'.format(pickle_dir), 'r') as f:
                medicine_dict = [ii.strip() for ii in f.readlines()]
            self.medicine_dict = medicine_dict
            ''' load medicine dict '''    
            medicine_noun = self.load_medicine_dict(medicine_dict)
            self.prop_dict = [*self.prop_dict, *medicine_noun]
            return

        medicine = pickle.load(open('{}medicine_dict.pkl'.format(pickle_dir), 'rb'))
        medicine_dict = medicine.split('\n')

        ctb8 = pickle.load(open('{}ctb8.pkl'.format(pickle_dir), 'rb'))
        msra = pickle.load(open('{}msra.pkl'.format(pickle_dir), 'rb'))
        weibo = pickle.load(open('{}weibo.pkl'.format(pickle_dir), 'rb'))
        other_dict = set([*ctb8, *msra, *weibo])

        medicine_dict = [ii for ii in medicine_dict if ii not in other_dict]
        if not repeat:
            medicine_blank = '   '.join(medicine_dict)
            medicine_dict = self.have_substring(medicine_dict, medicine_blank)
        medicine_dict = [*medicine_dict, *self.other_dict]
        print(len(medicine_dict))
        with open('{}medicine_dict.txt'.format(pickle_dir), 'w') as f:
            f.write('\n'.join(medicine_dict))

        self.medicine_dict = medicine_dict

    @jit
    def have_substring(self, medicine_dict, medicine_blank):
        ''' jit accelerate filter substring '''
        medicine_result = []
        distinguish_pattern = '.*性$|.*状$'
        for ww in medicine_dict:
            if len(re.findall(distinguish_pattern, ww)) or (len(ww) > 3 and medicine_blank.count(ww) == 1):
                medicine_result.append(ww)
        return medicine_result
    
    @jit
    def load_medicine_dict(self, medicine_dict):
        ''' jit load medicine dict '''
        result = []
        for ii in medicine_dict:
            if len(re.findall(self.medicine_pattern, ii)):
                result.append((ii, 'n'))
        return result

    def segment_test(self, types=2, mode=True, noThu=True):
        """
        word segmentation
        @param types: 0->pkuseg, 1->jieba, 2->jieba_v2, 3->thulac, 4->thulac_v2, 5->pkuseg_v2
        @param mode: True-> prediction, False-> test
        """
        origin_word = self.origin_word if mode else self.test_origin
        # origin_word = ['小儿脑性瘫痪']

        '''pkuseg'''
        seg = pkuseg.pkuseg(model_name='medicine')
        pos_pkuseg = [' '.join(seg.cut(ii)) for ii in origin_word]
        seg = pkuseg.pkuseg(model_name='medicine',
                            user_dict='{}medicine_dict.txt'.format(pickle_dir))
        pos_pkuseg_v2 = [' '.join(seg.cut(ii)) for ii in origin_word]

        '''jieba'''
        pos_jieba = [' '.join(jieba.cut(ii)) for ii in origin_word]
        jieba.load_userdict(self.medicine_dict)
        jieba.suggest_freq('$$_', True)
        for ii in self.del_dict:
            jieba.del_word(ii)
        pos_jieba_v2 = [' '.join(jieba.cut(ii)) for ii in origin_word]

        '''thulac'''
        if not noThu:
            thu1 = thulac.thulac(seg_only=True)
            pos_thulac = [thu1.cut(ii, text=True) for ii in origin_word]
            thu2 = thulac.thulac(
                seg_only=True, user_dict='%smedicine_dict.txt' % pickle_dir)
            pos_thulac_v2 = [thu2.cut(ii, text=True) for ii in origin_word]

        if not mode:
            print('Pkuseg\n', pos_pkuseg)
            self.evaluation_pos(pos_pkuseg, self.test_seg)
            print('Pkuseg & medicine\n', pos_pkuseg_v2)
            self.evaluation_pos(pos_pkuseg_v2, self.test_seg)
            print('Jieba\n', pos_jieba)
            self.evaluation_pos(pos_jieba, self.test_seg)
            print('Jieba & medicine\n', pos_jieba_v2)
            self.evaluation_pos(pos_jieba_v2, self.test_seg)
            if not noThu:
                print('Thulac\n', pos_thulac)
                self.evaluation_pos(pos_thulac, self.test_seg)
                print('Thulac & medicine\n', pos_thulac_v2)
                self.evaluation_pos(pos_thulac_v2, self.test_seg)
            print('Reference\n', self.test_seg)

        if not types:
            self.pos_word = pos_pkuseg
        elif types == 1:
            self.pos_word = pos_jieba
        elif types == 2:
            self.pos_word = pos_jieba_v2
        elif types == 3:
            self.pos_word = pos_thulac
        elif types == 4:
            self.pos_word = pos_thulac_v2
        elif types == 5:
            self.pos_word = pos_pkuseg_v2

    def segment_opt(self, mode=False, build_ref=False):
        """
        word segmentation optimization by using jieba
        """
        ref = self.train_seg if mode else self.test_seg
        pattern = '.*性.*'
        pattern_end = '.*性$'
        origin_word = self.origin_word if mode else self.test_origin
        if build_ref:
            jieba.load_userdict(self.medicine_dict)

        '''del dict'''
        jieba.suggest_freq('$$_', True)
        for ii in self.del_dict:
            jieba.del_word(ii)
        pos_jieba_v1 = [' '.join(jieba.cut(ii)) for ii in origin_word]
        self.evaluation_pos(pos_jieba_v1, ref)

        ''' pkuseg tag '''
        seg = pkuseg.pkuseg(model_name='medicine', postag=True)
        tag_pkuseg = [list(seg.cut(ii)) for ii in origin_word]
        tag_pkuseg = sum(tag_pkuseg, [])
        name_pku = [ii for ii, jj in tag_pkuseg if jj == 'nr']
        ns_pkuseg = [ii for ii, jj in tag_pkuseg if jj == 'ns']
        w_pkuseg = [ii for ii, jj in tag_pkuseg if jj == 'w']
        nt_pkuseg = [ii for ii, jj in tag_pkuseg if jj == 'nt']

        '''del pattern word'''
        tag_jieba_list = self.tag_rule(
            origin_word, ns_pkuseg, self.difference_word, name_pku)

        tag_jieba_str = [' '.join([str(jj) for jj in ii])
                         for ii in tag_jieba_list]
        tag_jieba = sum(tag_jieba_list, [])
        tag = set([jj for ii, jj in tag_jieba])

        name_jieba = [ii for ii, jj in tag_jieba if jj == 'nr']
        name_word = [ii for ii in name_jieba if ii in name_pku and len(
            ii) > 1 and ii not in self.name_exec]

        ns_jieba = [ii for ii, jj in tag_jieba if jj == 'ns']
        nt_jieba = [ii for ii, jj in tag_jieba if jj == 'nt']
        x_jieba = [ii for ii, jj in tag_jieba if jj == 'x']

        name_family = [ii[0] if ii[0] in self.family else ii[:2]
                       for ii in name_word]
        name_end = [ii[1:] if ii[0] in self.family else ii[2:]
                    for ii in name_word]
        end_word_two = [ii for ii, jj in tag_jieba if len(
            re.findall(pattern, ii)) and len(re.findall(pattern, ii)[0]) == 2]
        print(end_word_two)

        end_word_end = [ii for ii, jj in tag_jieba if len(re.findall(pattern_end, ii)) and len(
            re.findall(pattern_end, ii)[0]) > 2 and ii not in self.end_exec]
        end_word = [ii for ii, jj in tag_jieba if len(re.findall(pattern, ii)) and len(re.findall(
            pattern, ii)[0]) == 3 and ii not in self.end_exec and ii not in end_word_end]
        # for ii in end_word:
        #     jieba.del_word(ii)
        for (ii, jj) in self.prop_dict:
            jieba.add_word(ii, tag=jj)
        pos_jieba_v2 = [' '.join(jieba.cut(ii)) for ii in origin_word]

        self.evaluation_pos(pos_jieba_v2, ref)
        tag_jieba_list = self.tag_rule(
            origin_word, ns_pkuseg, self.difference_word, name_pku)
        tag_jieba_str = [' '.join([str(jj) for jj in ii])
                         for ii in tag_jieba_list]

        '''name & `$$_`'''
        pos_jieba_str = '||'.join(pos_jieba_v2)

        for ii in range(len(name_word)):
            pos_jieba_str = pos_jieba_str.replace(name_word[ii], '%s %s' % (
                name_family[ii], name_end[ii]))
        pos_jieba_str = pos_jieba_str.replace('$ $ _', '$$_')
        pos_jieba_str = pos_jieba_str.replace('$ $ _', '$$_')
        pos_jieba_v3 = pos_jieba_str.split('||')

        '''tag name & `$$_`'''
        tag_jieba_str_temp = '||'.join(tag_jieba_str)

        for ii in range(len(name_word)):
            tag_jieba_str_temp = tag_jieba_str_temp.replace('%s/nr' % name_word[ii], '%s/nr %s/nr' % (
                name_family[ii], name_end[ii]))
        tag_jieba_str_temp = tag_jieba_str_temp.replace('$/w $/w _/w', '$$_')

        '''<sub> <sup>'''
        tag_jieba_str_temp = tag_jieba_str_temp.replace('</w sup/e >/w ', '<sup>').replace(
            ' </w //w sup/e >/w', '</sup>').replace('</w sub/e >/w ', '<sub>').replace(' </w //w sub/e >/w', '</sub>')

        '''1. 2. '''
        for ii in range(10):
            tag_jieba_str_temp = tag_jieba_str_temp.replace(
                '%d./m' % ii, '%d/m ./w' % ii)
        for ii in end_word_end:
            tag_jieba_str_temp = tag_jieba_str_temp.replace(
                '%s/n' % ii, '%s/n 性/k' % ii[:-1])

        tag_jieba_v3 = tag_jieba_str_temp.split('||')

        self.evaluation_pos(pos_jieba_v3, ref)
        if mode:
            if build_ref:
                with open('%sresult_ref' % result_dir, 'w') as f:
                    f.write('\n'.join(pos_jieba_v3))
                with open('%sresult_ref_tag' % result_dir, 'w') as f:
                    f.write('\n'.join(tag_jieba_v3))
            else:
                with open('%s' % seg_result_dir, 'w') as f:
                    f.write('\n'.join(pos_jieba_v3))
                with open('%s' % pos_result_dir, 'w') as f:
                    f.write('\n'.join(tag_jieba_v3))

        print(name_word)
        print(set(name_jieba))
        print(set(name_pku))
        print(name_family)
        print(name_end)

        print(ns_jieba)
        print(ns_pkuseg)
        print(nt_jieba)
        print(nt_pkuseg)
        print(set(x_jieba))
        print(set(w_pkuseg))
        print(sorted(tag))
        print(set(end_word))
        print(set(end_word_two))
        print(set(end_word_end))
        if not mode:
            print(tag_jieba)
            print(pos_jieba_v3)
            print(self.test_seg)

    def pos_opt(self, mode=False, build_ref=False, origin_word=None, output_file=None):
        """
        pos tag optimization by using jieba
        """
        ref = self.train_tag if mode else self.test_tag
        pattern = '.*性.*|.*状'
        pattern_end = '.*性$|.*状'
        if origin_word is None:
            origin_word = self.origin_word if mode else self.test_origin
        if build_ref:
            jieba.load_userdict(self.medicine_dict)

        '''del dict'''
        jieba.suggest_freq('$$_', True)
        for ii in self.del_dict:
            jieba.del_word(ii)

        ''' pkuseg tag '''
        seg = pkuseg.pkuseg(model_name='medicine', postag=True)
        tag_pkuseg = [list(seg.cut(ii)) for ii in origin_word]
        tag_pkuseg = sum(tag_pkuseg, [])
        name_pku = [ii for ii, jj in tag_pkuseg if jj == 'nr']
        ns_pkuseg = [ii for ii, jj in tag_pkuseg if jj == 'ns']
        w_pkuseg = [ii for ii, jj in tag_pkuseg if jj == 'w']
        nt_pkuseg = [ii for ii, jj in tag_pkuseg if jj == 'nt']

        '''del pattern word'''
        tag_jieba_list = self.tag_rule(
            origin_word, ns_pkuseg, self.difference_word, name_pku)

        tag_jieba_str = [' '.join([str(jj) for jj in ii])
                         for ii in tag_jieba_list]
        tag_jieba = sum(tag_jieba_list, [])
        tag = set([jj for ii, jj in tag_jieba])

        name_word = [ii for ii, jj in tag_jieba if jj == 'nr']

        ns_jieba = [ii for ii, jj in tag_jieba if jj == 'ns']
        nt_jieba = [ii for ii, jj in tag_jieba if jj == 'nt']
        x_jieba = [ii for ii, jj in tag_jieba if jj == 'x']

        name_family = [ii[0] if ii[0] in self.family else ii[:2]
                       for ii in name_word]
        name_end = [ii[1:] if ii[0] in self.family else ii[2:]
                    for ii in name_word]
        end_word_two = [ii for ii, jj in tag_jieba if len(
            re.findall(pattern, ii)) and len(re.findall(pattern, ii)[0]) == 2]
        print(end_word_two)

        end_word_end = [ii for ii, jj in tag_jieba if len(re.findall(pattern_end, ii)) and len(
            re.findall(pattern_end, ii)[0]) > 2 and ii not in self.end_exec]
        end_word = [ii for ii, jj in tag_jieba if len(re.findall(pattern, ii)) and len(re.findall(
            pattern, ii)[0]) == 3 and ii not in self.end_exec and ii not in end_word_end]
        # for ii in end_word:
        #     jieba.del_word(ii)
        self.prop_dict = [*self.prop_dict, *[(ii, 'b') for ii in end_word_end]]
        for (ii, jj) in self.prop_dict:
            jieba.add_word(ii, tag=jj)
        pos_jieba_v2 = [' '.join(jieba.cut(ii)) for ii in origin_word]

        tag_jieba_list = self.tag_rule(
            origin_word, ns_pkuseg, self.difference_word, name_pku)
        tag_jieba_str = [' '.join([str(jj) for jj in ii])
                         for ii in tag_jieba_list]

        '''name & `$$_`'''
        pos_jieba_str = '||'.join(pos_jieba_v2)

        for ii in range(len(name_word)):
            pos_jieba_str = pos_jieba_str.replace(name_word[ii], '%s %s' % (
                name_family[ii], name_end[ii]))
        pos_jieba_str = pos_jieba_str.replace('$ $ _', '$$_')
        pos_jieba_str = pos_jieba_str.replace('$ $ _', '$$_')
        pos_jieba_str = pos_jieba_str.replace('\\\$\$_', '$$_')
        pos_jieba_v3 = pos_jieba_str.split('||')

        '''tag name & `$$_`'''
        tag_jieba_str_temp = '||'.join(tag_jieba_str)

        for ii in range(len(name_word)):
            tag_jieba_str_temp = tag_jieba_str_temp.replace('%s/nr' % name_word[ii], '%s/nr %s/nr' % (
                name_family[ii], name_end[ii]))
        tag_jieba_str_temp = tag_jieba_str_temp.replace('$/w $/w _/w', '$$_')

        '''<sub> <sup>'''
        tag_jieba_str_temp = tag_jieba_str_temp.replace('</w sup/nx >/w ', '<sup>').replace(
            ' </w //w sup/nx >/w', '</sup>').replace('</w sub/nx >/w ', '<sub>').replace(' </w //w sub/nx >/w', '</sub>')

        '''1. 2. '''
        for ii in range(30):
            tag_jieba_str_temp = tag_jieba_str_temp.replace(
                '%d./w' % ii, '%d/m ./w' % ii)
        # for ii in end_word_end:
        #     tag_jieba_str_temp = tag_jieba_str_temp.replace(
        #         '%s/n' % ii, '%s/n 性/k' % ii[:-1])
        # tag_jieba_str_temp = tag_jieba_str_temp.replace('小管囊性/n', '小管 ')

        '''/n 性/k'''
        # tag_jieba_str_temp = re.sub('/. 性/n', '/n 性/k', tag_jieba_str_temp)
        # tag_jieba_str_temp = re.sub('/.. 性/n', '/n 性/k', tag_jieba_str_temp)
        tag_jieba_str_temp = re.sub('囊/Ng 腔/.', '囊腔/n', tag_jieba_str_temp)
        tag_jieba_str_temp = re.sub('囊/Ng 炎/.', '囊炎/n', tag_jieba_str_temp).replace('囊/Ng ', '囊')
        tag_jieba_str_temp = re.sub('/. 性病变/n', '/n 性/k 病变/n', tag_jieba_str_temp)
        tag_jieba_str_temp = tag_jieba_str_temp.replace('伴眼/v', '伴/v 眼/n').replace('检查/vn', '检查/n')

        tag_jieba_v3 = tag_jieba_str_temp.split('||')

        self.evaluation_pos(tag_jieba_v3, ref)
        if mode:
            if build_ref:
                with open('%sresult_ref' % result_dir, 'w') as f:
                    f.write('\n'.join(pos_jieba_v3))
                with open('%sresult_ref_tag' % result_dir, 'w') as f:
                    f.write('\n'.join(tag_jieba_v3))
            else:
                if output_file is None:
                    with open('%s' % seg_result_dir, 'w') as f:
                        f.write('\n'.join(pos_jieba_v3))
                    with open('%s' % pos_result_dir, 'w') as f:
                        f.write('\n'.join(tag_jieba_v3))
                else:
                    with open('%s%s' % (result_dir, output_file), 'w') as f:
                        f.write('\n'.join(tag_jieba_v3))

        print(name_word)
        # print(set(name_jieba))
        print(set(name_pku))
        print(name_family)
        print(name_end)

        print(ns_jieba)
        print(ns_pkuseg)
        print(nt_jieba)
        print(nt_pkuseg)
        print(set(x_jieba))
        print(set(w_pkuseg))
        print(sorted(tag))
        print(set(end_word))
        print(set(end_word_two))
        print(set(end_word_end))
        if not mode:
            print(tag_jieba)
            print(pos_jieba_v3)
            print(self.test_seg)

    def ner_opt(self):
        '''ner optimization'''
        self.wait_dis = []
        self.replace_dis = []
        self.wait_body = []
        self.replace_body = []
        with open('%s' % pos_result_dir, 'r') as f:
            origin_ner = [ii.strip() for ii in f.readlines()]

        origin_ner_list = sum([('%s $$_' % ii).split()
                               for ii in origin_ner], [])
        origin_str = '||'.join(self.origin_word)
        origin_ner_str = '||'.join(origin_ner)
        self.origin_str = origin_str
        self.origin_ner_list = origin_ner_list
        self.origin_ner_str = origin_ner_str

        stop_dis = ['c', 'p', 'v', 'w', 'u', 'd', 'q', 'f', 'r', 't']
        stop_body = ['c', 'p', 'v', 'w', 'u', 'd', 'k', 'a', 'r', 'i', 't', 'b', 'l', 'z', 'n', 'q', 'f']
        ner_need = ['disease', 'symptom', 'test', 'treatment']
        for ii in ner_need:
            self.extract_ner(ii, stop_dis)
        self.extract_ner('body', stop_body)

        wait_dis = [(jj, ii) for ii, jj in enumerate(self.wait_dis)]
        replace_dis = {ii: jj for ii, jj in enumerate(self.replace_dis)}
        wait_dis = sorted(wait_dis, key=lambda ii: len(ii[0]), reverse=True)
        print(wait_dis)
        print(replace_dis)
        for (ii, jj) in wait_dis:
            # if ii == ''
            origin_ner_str = origin_ner_str.replace(ii, '@%d@@%d@@@%d@@@@'%(jj,jj,jj))
        for (_, jj) in wait_dis:
           origin_ner_str = origin_ner_str.replace('@%d@@%d@@@%d@@@@'%(jj,jj,jj),replace_dis[jj])

        ''' longest-submatch '''
        wait_body = [(jj, ii) for ii, jj in enumerate(self.wait_body)]
        replace_body = {ii: jj for ii, jj in enumerate(self.replace_body)}
        wait_body = sorted(wait_body, key=lambda ii: len(ii[0]), reverse=True)
        print(wait_body)
        print(replace_body)
        for (ii, jj) in wait_body:
            origin_ner_str = origin_ner_str.replace(ii, '@@@@%d@@@%d@@%d@'%(jj,jj,jj))
        for (_, jj) in wait_body:
           origin_ner_str = origin_ner_str.replace('@@@@%d@@@%d@@%d@'%(jj,jj,jj),replace_body[jj])  
        # origin_ner_str = origin_ner_str.replace(']sym 、/w [', ' 、/w ')
        # origin_ner_str = origin_ner_str.replace(']sym 及/c [', ' 及/c ')
        # origin_ner_str = origin_ner_str.replace(']sym 和/c [', ' 和/c ')
        # origin_ner_str = origin_ner_str.replace(']sym 或/c [', ' 或/c ')

        '''other'''
        origin_ner_str = origin_ner_str.replace('[肺炎/n]dis ', '肺炎/n ')

        '''nt'''
        origin_ner_str = origin_ner_str.replace('国际/n 小儿/n 肿瘤/n 协会/n', '[国际/n 小儿/n 肿瘤/n 协会/n]nt')
        origin_ner_str = origin_ner_str.replace('北京医科大学/nt 出版社/n', '[北京/ns 医科/n 大学/n 出版社/n]nt')
        origin_ner_str = origin_ner_str.replace('北京/ns 儿童医院/nt', '[北京/ns 儿童/n 医院/n]nt')
        ner_result = origin_ner_str.split('||')
        ner_result = ['%d %s'%(ii+1, jj) for ii,jj in enumerate(ner_result)]
        with open('%s' % ner_result_dir, 'w') as f:
            f.write('\n'.join(ner_result))

    def extract_ner(self, types, stop_list):
        '''extract ner'''
        origin_ner_list = self.origin_ner_list
        with open('%s%s.txt' % (data_dir, types), 'r') as f:
            origin_extract = [ii.strip() for ii in f.readlines()]
        self.pos_opt(mode=True, origin_word=origin_extract,
                     output_file=types+'_pos')
        with open('%s%s_pos' % (result_dir, types), 'r') as f:
            origin_extract = [ii.strip() for ii in f.readlines()]

        extract_list = [
            ii for ii in origin_extract if ii in self.origin_ner_str]
        print('matching %d' % len(extract_list))
        pattern_end = {ii.split()[-1]: (len(ii.split()), ii)
                       for ii in extract_list if len(ii.split())}

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
                        last_flag = origin_ner_list[last_index].split(
                            '/')[1][0]
                    except:
                        last_flag = stop_list[0]
                        # print(origin_ner_list[last_index])
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
        # print(word_str, pattern_str)
        return len(re.findall('.*%s' % pattern_str, word_str)) or len(re.findall('.*%s' % word_str, pattern_str))

    def pre_ner(self, origin_list, types):
        ''' prepare ner '''
        temp_origin = ' '.join(origin_list)
        if temp_origin in self.ner_error:
            return
        if types == 'bod':
            self.wait_body.append('%s' % temp_origin)
            self.replace_body.append('[%s]%s' % (temp_origin, types))
            return
        self.wait_dis.append('%s' % temp_origin)
        self.replace_dis.append('[%s]%s' % (temp_origin, types))

    def tag_rule(self, origin_data, ns_pkuseg, end_word_two=None, name_pku=[]):
        '''tag rule'''
        tag_jieba_list = []
        eng_pattern = '[a-zA-Z]+'
        error = []
        seg = pkuseg.pkuseg(model_name='medicine', postag=True)

        for mm in origin_data:
            temp_tag = []
            for ii in pseg.cut(mm):
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
                            nn = [tt for tt in pseg.cut(ii.word)][0]
                            nn.word = jj
                            nn.flag = kk
                            temp_tag.append(nn)
                            print(111, nn)
                        ii.word = pkuseg_pos_tmp[-1][0] 
                        ii.flag = pkuseg_pos_tmp[-1][1]
                        # print(222, ii)
                        # /ii.flag = 'n'
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
        self.error = set(error)
        return tag_jieba_list

    def tag_data(self, types=2, mode=True):
        """
        part-of-speech tagging
        """
        origin_word = self.origin_word if mode else self.test_origin
        '''pkuseg'''
        seg = pkuseg.pkuseg(model_name='medicine', postag=True)
        tag_pkuseg = [' '.join(seg.cut(ii)) for ii in origin_word]

        '''jieba'''
        posseg = jieba.posseg
        tag_jieba = [' '.join(posseg.cut(ii)) for ii in origin_word]
        jieba.load_userdict(self.medicine_dict)
        tag_jieba_v2 = [' '.join(posseg.cut(ii)) for ii in origin_word]

        '''thulac'''
        thu1 = thulac.thulac()
        tag_thulac = [thu1.cut(ii, text=True) for ii in origin_word]
        thu2 = thulac.thulac(user_dict='%smedicine_dict.txt' % pickle_dir)
        tag_thulac_v2 = [thu2.cut(ii, text=True) for ii in origin_word]

        if not mode:
            print('Pkuseg\n', tag_pkuseg)
            # self.evaluation_pos(pos_pkuseg, self.test_seg)
            print('Jieba\n', tag_pkuseg)
            # self.evaluation_pos(pos_jieba, self.test_seg)
            print('Jieba & medicine\n', tag_jieba_v2)
            # self.evaluation_pos(pos_jieba_v2, self.test_seg)
            print('Thulac\n', tag_thulac)
            # self.evaluation_pos(pos_thulac, self.test_seg)
            print('Thulac & medicine\n', tag_thulac_v2)
            # self.evaluation_pos(pos_thulac_v2, self.test_seg)
            print('Reference\n', self.test_tag)

    def load_file(self, file_name):
        '''load file'''
        with open(file_name, 'r') as f:
            if 'raw' in raw_dir:
                origin = [ii.strip().split(None, 1)[1] for ii in f.readlines()]
            else:
                origin = [ii.strip() for ii in f.readlines()]
        return origin

    def evaluation_pos(self, predict, result):
        '''evaluation pos'''
        predict = [ii.split('/')[0] for ii in predict]
        result = [ii.split('/')[0] for ii in result]
        predict = ' '.join([ii.strip() for ii in predict])
        result = ' '.join([ii.strip() for ii in result])
        # print(predict.split())
        # print(result.split())

        predict_index = self.wordlist2tuple(predict)
        result_index = self.wordlist2tuple(result)
        result_str = result.replace(
            ' ', '').replace(' ', '').replace(' ', '')
        error_list = [ii for ii in result_index if ii not in predict_index]
        if len(result_index) and len(predict_index):
            for (ii, jj) in error_list:
                if jj < len(result_index):
                    print(ii, result_str[ii:jj])
        true_num = len(result_index) - len(error_list)
        r = true_num / len(result_index) if len(result_index) else 0
        p = true_num / len(predict_index) if len(predict_index) else 0
        f1 = (2 * r * p) / (r + p) if (r + r) else 0
        print('R: {:.2f}, P: {:.2f}, F1: {:.2f}'.format(r*100, p*100, f1*100))
        return r, p, f1

    def wordlist2tuple(self, wordlist):
        '''wordlist to index tuple'''
        word_index = []
        begin_index, end_index = 0, 0
        for ii in wordlist.split():
            begin_index = end_index
            end_index += len(ii)
            word_index.append((begin_index, end_index))
        return word_index


if __name__ == "__main__":
    pp = pretreat()
    pp.pos_opt(mode=True)
    pp.ner_opt()
