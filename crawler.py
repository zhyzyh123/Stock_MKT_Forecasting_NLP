# -*-coding:utf-8 -*-

import urllib2
import re
import xlwt

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('sheet', cell_overwrite_ok=True)
num = 800

# use j to change the target page in the url
for j in range(1, num * 20 + 1):
    if j % num == 1:
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('sheet', cell_overwrite_ok=True)
    print(j)
    url = 'http://guba.eastmoney.com/list,zssh000001,f_' + str(j) + '.html'
    try:
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        content = response.read().decode('utf-8')
        pattern = re.compile('<span class.*?title=(.*?)>', re.S)
        title = re.findall(pattern, content)
        pattern = re.compile('data-popper.*?<span class.*?>(.*?)</span>.*?articleh normal_post', re.S)
        time = re.findall(pattern, content)
        l = min(len(title), len(time))
        for i in range(0, l):
            titleans = title[i + 1].strip('"')
            sheet.write(((j - 1) % num) * l + i, 0, titleans)
            fabiaotime = time[i]
            sheet.write(((j - 1) % num) * l + i, 1, fabiaotime)

    except urllib2.URLError, e:
        if hasattr(e, "code"):
            print(e.code)

        if hasattr(e, "reason"):
            print(e.reason)

    if j % num == 0:
        book.save('data/raw_data_more/' + str(j / num) + '_zssh.xls')
