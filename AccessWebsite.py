#!/usr/bin/env python
# -*-coding:utf-8 -*-
import urllib
import urllib2
import re
import xlwt

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('sheet', cell_overwrite_ok=True)

# j的作用是对url不断进行修改，翻页
for j in range(1, 1500):
    if j % 200 == 1:
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('sheet' ,cell_overwrite_ok=True)
    print(j)
    url = 'http://guba.eastmoney.com/list,zssh000001,f_' + str(j) + '.html'
    try:
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        content = response.read().decode('utf-8')
        pattern = re.compile('<span class.*?title=(.*?)>', re.S)
        title = re.findall(pattern, content)
        pattern = re.compile('data-popper.*?<font>(.*?)</font>', re.S)
        author = re.findall(pattern, content)
        pattern = re.compile('data-popper.*?<span class.*?>(.*?)</span>.*?articleh normal_post', re.S)
        time = re.findall(pattern, content)
        pattern = re.compile('articleh.*?<span.*?>(.*?)</span>', re.S)
        num = re.findall(pattern, content)
        l = min(len(title), len(author), len(time), len(num))
        for i in range(0, l):
            titleans = title[i + 1].strip('"')
            sheet.write(((j - 1)%200) * l + i, 0, titleans)
            authorans = author[i]
            sheet.write(((j - 1)%200) * l + i, 1, authorans)
            fabiaotime = time[i]
            sheet.write(((j - 1)%200) * l + i, 2, fabiaotime)
            yuedu = num[i][0]
            sheet.write(((j - 1)%200) * l + i, 3, yuedu)
            pinglun = num[i][1]
            sheet.write(((j - 1)%200) * l + i, 4, pinglun)

    except urllib2.URLError, e:
        if hasattr(e, "code"):
            print(e.code)

        if hasattr(e, "reason"):
            print(e.reason)

    if j % 200 == 0:
        book.save(str(j / 200) + '_zssh.xls')
