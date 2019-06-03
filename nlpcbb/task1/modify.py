'''
@Author: gunjianpan
@Date:   2019-04-30 22:01:01
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-30 23:49:55
'''

import re
from bs4 import BeautifulSoup

student_id = 1
data_dir = 'data/'
raw_dir = '{}raw_{}.txt'.format(data_dir, student_id)
pickle_dir = 'pickle/'
result_dir = 'result/'
origin_path = '{}3rd_{}_6.txt'.format(result_dir, student_id)
result_path = '{}3rd_{}.txt'.format(result_dir, student_id)
check_path = '{}check_1_1st.html'.format(data_dir)


class Modify:
    ''' modify by using check html '''

    def __init__(self):
        self.load()

    def load(self):
        ''' load origin text & check result '''
        with open(origin_path, 'r') as f:
            self.origin_text = [ii.strip() for ii in f.readlines()]
        with open(check_path, 'r') as f:
            refer_html = BeautifulSoup(f.read(), 'html.parser')
        refer, last_row = {}, 0
        for ii in refer_html.findAll('font')[1:-1]:
            color, text = ii['color'], ii.text
            if color == 'blue':
                last_row = int(re.findall('第(.*?)行', text)[0])
                refer[last_row] = []
            elif color == 'purple':
                pos = re.findall('“(.*?)”', text)
                pos[1] = '{}/{}'.format(pos[0].split('/')[0], pos[1])
                refer[last_row].append([0, *pos])
            elif color == 'olive':
                refer[last_row].append([1, *re.findall('“(.*?)”', text)])
            elif color == 'red':
                print(last_row, text)
            elif color == 'teal':
                pos = re.findall('“(.*?)”', text)
                pos[1] = '{}]{}'.format(pos[0].split(']')[0], pos[1])
                refer[last_row].append([2, *pos])
        self.refer = refer

    def change_info(self):
        ''' change info base on refer '''
        result = self.origin_text
        for ii, jj in enumerate(result):
            if not ii + 1 in self.refer:
                continue
            info_list = self.refer[ii + 1]
            for kk in info_list:
                if kk[0] == 1:
                    if len(kk) == 2:
                        try:
                            origin_str = re.findall(kk[1].replace(
                                ' ', '/[a-zA-Z]{1,2} '), jj)[0]
                            replace_str = kk[1].replace(' ', '')
                        except:
                            print(jj, '\n', kk[1])
                    elif len(kk) == 3:
                        replace_list = kk[2].split()
                        replace_str = '{}/n {}'.format(
                            replace_list[0], replace_list[1])
                        origin_str = kk[1]
                elif kk[0] == 2:

                else:
                    origin_str = kk[1]
                    replace_str = kk[2]
                result[ii] = result[ii].replace(
                    origin_str, replace_str).replace('/az', '/a').replace('/nn', '/n').replace('/nd', '/n').replace('/vz', '/v')
        with open(result_path, 'w') as f:
            f.write('\n'.join(result))
