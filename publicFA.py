# 通用函数
from SQL import MSSQL
import os
import numpy as np
# import Reseroperation
import datetime


def get_sql():  # 调度库
    r = []
    with open('sql.txt', 'r') as f:
        for line in f:
            r.append(list(line.strip('\n').split(','))[0])
    mssql = MSSQL(host="%s" % r[0], port=int(r[1]), user="%s" % r[2], pwd="%s" % r[3], db="%s" % r[4])
    return mssql


def get_ybsql():  # 读预报库
    r = []
    with open('sqlyb.txt', 'r') as f:
        for line in f:
            r.append(list(line.strip('\n').split(','))[0])
    ybsql = MSSQL(host="%s" % r[0], port=int(r[1]), user="%s" % r[2], pwd="%s" % r[3], db="%s" % r[4])
    return ybsql

# 读库用
# 数据库连接通用函数---2023.2.8徐孙钰
# 配置文件：sql_list.yaml
# 函数输入与上述文件中的key一致即可
def get_sql_unique(system):
    import yaml
    from SQL import MSSQL
    with open("sql_list.yaml", "r") as ymlfile:
        password = yaml.safe_load(ymlfile)
    unihost = password[system]["host"]
    uniport = password[system]["port"]
    uniuser = password[system]["user"]
    unipassword = password[system]["password"]
    unidb = password[system]["db"]
    mssql = MSSQL(host="%s" % unihost, port=int(uniport), user="%s" % uniuser, pwd="%s" % unipassword, db="%s" % unidb)
    return mssql


# 写库用
# 数据库连接通用函数---2023.2.8徐孙钰
def get_yconnect_unique(system):
    import yaml
    with open("sql_list.yaml", "r") as ymlfile:
        password = yaml.safe_load(ymlfile)
    unihost = password[system]["host"]
    uniport = password[system]["port"]
    uniuser = password[system]["user"]
    unipassword = password[system]["password"]
    unidb = password[system]["db"]
    from sqlalchemy import create_engine
    yconnect = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8' % (
        uniuser, unipassword, unihost, uniport, unidb))  # root:123456@127.0.0.1:3306/qtj
    return yconnect


def get_pubSql():  # 读公共数据库 主要是泄流能力数据统一
    r = []
    with open('sql.txt', 'r') as f:
        for line in f:
            r.append(list(line.strip('\n').split(','))[0])
    mssql = MSSQL(host="%s" % r[5], port=int(r[6]), user="%s" % r[7], pwd="%s" % r[8], db="%s" % r[9])
    return mssql


def get_yconnect():
    r = []
    with open('sql.txt', 'r') as f:
        for line in f:
            r.append(list(line.strip('\n').split(','))[0])
    from sqlalchemy import create_engine
    # yconnect = create_engine('mssql+pymssql://%s:%s@%s/%s?charset=utf8' % (r[1], r[2], r[0], r[3]))
    yconnect = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8' % (
        r[2], r[3], r[0], r[1], r[4]))  # root:123456@127.0.0.1:3306/qtj
    return yconnect


def get_name(systemid):  # 读取了所有的水库和防洪点 到时候其他子片区可能需要额外处理
    # 通过数据库读取水库名称
    mssql = get_sql_unique(systemid)
    result = mssql.ExecQuery(
        "SELECT Name FROM Kuname where ID is not NULL and Type ='1' order by ID" )
    nameSK = []
    for (row,) in result:
        nameSK.append(row)
    result = mssql.ExecQuery(
        "SELECT Name FROM Kuname where ID is not NULL and Type ='2' order by ID" )
    nameFHD = []
    for (row,) in result:
        nameFHD.append(row)
    return nameSK, nameFHD


def reget_name(SYSTEM):  # zhao修改
    # 通过数据库读取水库名称
    mssql = get_sql()
    result = mssql.ExecQuery("SELECT RES_CODE FROM uuid where ID is not NULL and SYSTEM='%s' order by ID" % SYSTEM)
    nameSK = []
    for (row,) in result:
        nameSK.append(row)
    # 防洪点名称
    if SYSTEM == '衢州以上子系统':
        nameFHD = ["江山", "常山", "衢州"]  # 之前采用站码，但和前端沟通后，直接采用中文
    elif SYSTEM == '兰溪以上子系统':
        nameFHD = ["龙游", "兰溪"]  # 计算兰溪时 需额外加衢州来水。此外，在ybbasin_info中顺序为 龙游、衢州、金华 兰溪 因此这儿定顺序 龙游排在衢州前
    elif SYSTEM == '金华以上子系统':
        nameFHD = ["东阳", "金华"]
    elif SYSTEM == '桐庐以上子系统':
        nameFHD = ["梅城", "桐庐"]  # ybbasin_info中顺序为 梅城、兰溪、桐庐小区间 因此这儿定顺序
    elif SYSTEM == '诸暨以上子系统':
        nameFHD = ["诸暨"]
    elif SYSTEM == '上虞以上子系统':
        nameFHD = ["嵊州", "上虞"]
    elif SYSTEM == "五座大型水库":
        nameFHD = ["江山", "常山", "衢州", "东阳", "金华", "龙游", "兰溪", "梅城", "桐庐", "诸暨", "嵊州", "上虞"]
    else:
        nameFHD = []
    return nameSK, nameFHD


# 演进数据需求
def fhdget_name(SYSTEM):  # zhao修改
    # 防洪点名称
    if SYSTEM == '衢州以上子系统':
        nameFHD = ["江山", "常山", "衢州"]  # 之前采用站码，但和前端沟通后，直接采用中文
        nameFHDcd = ['70104650', '70100350', '70100500']
    elif SYSTEM == '兰溪以上子系统':
        nameFHD = ["龙游", "兰溪"]  # 计算兰溪时 需额外加衢州来水。此外，在ybbasin_info中顺序为 龙游、衢州、金华 兰溪 因此这儿定顺序 龙游排在衢州前
        nameFHDcd = ['70108050', '70100900']
    elif SYSTEM == '金华以上子系统':
        nameFHD = ["东阳", "金华"]
        nameFHDcd = ['70108050', '70108400']
    elif SYSTEM == '桐庐以上子系统':
        nameFHD = ["梅城"]  # ybbasin_info中顺序为 梅城、兰溪、桐庐小区间 因此这儿定顺序
        nameFHDcd = ['70112400']
    elif SYSTEM == "五座大型水库":
        nameFHD = ["江山", "常山", "衢州", "东阳", "金华", "龙游", "兰溪", "梅城"]
        nameFHDcd = ['70104650', '70100350', '70100500', '70108050', '70108400', '70110050', '70100900',
                     '70112400']  # 龙游暂时瞎写
    else:
        nameFHD = []
    return nameFHDcd, nameFHD


# 水库基础信息读取（水库的常用参数、特征曲线、该水库尾水水位流量关系10003）

# TODO：Warning：本来要读取的第三个系列是该水库对应最临近防洪点断面水位流量关系10007，但库群中很难满足每个水库都有自己的防洪点，所以这块暂时用
# 尾水水位流量关系占位，目前水库的zqtemp对应的是尾水水位流量关系，而不是临近断面的水位流量关系，故用此套数据调用单库模型得到的水库方案详情resw中
# qcMsjg对应的水位zz是假的，如果要为真，要把zqtemp变成读该水库对应10007得到的数据。

# cs[区间缩放系数,最大发电流量pp1,发电装机fd,马斯京根nn,xe,ke]
# tez[水位,库容,水位,耗水率,水位,总泄流能力] 分别对应三根曲线：水库水库库容曲线、耗水率曲线、水库泄流能力曲线
# zqtemp[水位,流量] 该水库尾水水位流量关系10003 (本应读取该水库对应最临近防洪点断面水位流量关系10007)
# 水库泄流能力曲线和数字公司共用一套，由数字公司维护，所以本应该从数字公司数据库中读取，但是目前没有，暂时从我们自己的数据库中读取。
# 故函数参数中的uuid和publiSql是用来从数字公司数据库读泄流能力的，现在读自己的数据库没用到这两个参数


def PreRead(id, mssql):
    # 水位库容曲线
    # zv = mssql.ExecQuery("select M_Z,M_DATA from charadata where SYSTEM = '%s' and SK_ID='%s' and CH_ID=10001 order by M_Z" % (SYSTEM, id))
    zv = mssql.ExecQuery(
        "select M_Z,M_DATA from charadata where SK_ID='%s' and CH_ID=10001 order by M_Z" % id)
    z1 = np.array(zv)[:, 0]
    v = np.array(zv)[:, 1]
    # 耗水率曲线
    # zhs = mssql.ExecQuery("select M_Z,M_DATA from charadata where SYSTEM = '%s' and SK_ID='%s' and CH_ID=10005 order by M_Z" % (SYSTEM, id))
    # z2 = np.array(zhs)[:, 0]
    # hs = np.array(zhs)[:, 1]
    # 泄流能力曲线
    # qxltemp = mssql.ExecQuery(
    #     "select M_Z,M_DATA from charadata where SYSTEM = '%s' and SK_ID='%s' and CH_ID=10004 order by M_Z" % (
    #         SYSTEM, id))
    qxltemp = mssql.ExecQuery(
        "select M_Z,M_DATA from charadata where SK_ID='%s' and CH_ID=10004 order by M_Z" % id)
    z3 = np.array(qxltemp)[:, 0]
    qxl = np.array(qxltemp)[:, 1]
    # 本应从数字公司数据库读，公用一套泄流数据
    # qxltemp = publicSql.ExecQuery("select water_level,flood_discharge from flood_ability_curve where type = 'all' and res_code='%s' order by id" % (uuid))
    # z3 = np.array(qxltemp)[:, 0]
    # qxl = np.array(qxltemp)[:, 1]

    # 常用数据
    # 默认顺序 1nn,2 Xe,3Ke 4qm1 5nnum1 6jizu1 7qm2 8nnum2 9jizu2 10chuli1 11chuli2 12ny1 13ny2 14xiayouZQ 15diaoduZQ 16pp1 17fd
    cstemp = mssql.ExecQuery("select data from inikuqun where code='%s'" % id)
    cstemp = np.array(cstemp)
    cs = np.zeros(14)
    cs[0] = 0.1  # 目前缩放系数都设置为0.1
    cs[0] = cstemp[15]  # pp1 发电流量
    cs[2] = cstemp[16]  # fd 发电装机
    cs[3] = cstemp[0]  # nn
    cs[4] = cstemp[1]  # xe
    cs[5] = cstemp[2]  # kebuhsi
    cs[6] = cstemp[6]  # qm2
    cs[7] = cstemp[7]  # nnum2
    cs[8] = cstemp[8]  # jizu2
    cs[9] = cstemp[9]  # chuli1
    cs[10] = cstemp[10]  # chuli2
    cs[11] = cstemp[3]  # qm1
    cs[12] = cstemp[4]  # nnum1
    cs[13] = cstemp[5]  # jizu1

    # 读取闸门开度曲线z~kaidu~q关系
    ZMKDtemp = mssql.ExecQuery(
        "select Zhamen,M_Z,kaidu1,kaidu2,kaidu3,kaidu4,kaidu5,kaidu6 from kaidudata where SK_ID='%s' order by M_Z" %
        id)
    ZMKD = np.array(ZMKDtemp)

    # 防洪点Z-Q关系
    # 36江山37常山38衢州39龙游40东阳41金华42兰溪43兰溪44桐庐45诸暨46闻堰47嵊州48上虞49杭州
    # 水库：尾水水位流量关系
    # 水库：模糊的下游水位流量关系 防洪点：防洪点水位流量关系
    xy_ID = mssql.ExecQuery("select next_code from kuname where ID = '%s' "% id)
    xy_ID = np.array(xy_ID)[0][0]
    zqtemp1 = mssql.ExecQuery(
        "select M_Z,M_DATA from charadata where CH_ID=10003 and SK_ID = '%s' order by M_Z"% id)
    zqtemp = np.array(zqtemp1)

    tez = []
    tez.append(z1)
    tez.append(v)
    # tez.append(z2)
    # tez.append(hs)
    tez.append(z3)
    tez.append(qxl)
    return tez, cs, zqtemp, ZMKD
    # return tez, cs, ZMKD


# 防洪点基础信息读取（防洪点马斯京根参数、水位流量关系）
# cs[区间缩放系数,最大发电流量pp1,发电装机fd,马斯京根nn,xe,ke],其实防洪点前三个值没有，这里用0占位，防洪点有用的只有后三个马斯京根参数
# zqtemp[水位,流量] 防洪点断面水位流量关系
##常用数据 顺序 nn Xe Ke
def PreReadFHD(id, mssql, SYSTEM):
    cstemp = np.zeros(6)
    result = mssql.ExecQuery("SELECT data FROM inikuqun where SYSTEM='%s' and code='%s'" % (
        SYSTEM, id))  # np.loadtxt(dir + '\\data\\' + str(i) + '\\cysj.dat')
    kk = 3
    for (row,) in result:
        cstemp[kk] = row
        kk += 1
    cs = cstemp.tolist()
    # zq水库防洪点水位流量关系
    result = mssql.ExecQuery("SELECT M_Z,M_DATA FROM fhd_charadata where SYSTEM='%s' and FHD_ID='%s' order by M_Z" % (
        SYSTEM, id))  # 防洪点的fhd_charadata只有水位流量关系10007
    zqtemp = np.array(result)
    # zq = zqtemp.tolist()
    return cs, zqtemp

def PreRead_FHD(id, mssql):

    # 常用数据
    # 默认顺序 1nn,2 Xe,3Ke 4qm1 5nnum1 6jizu1 7qm2 8nnum2 9jizu2 10chuli1 11chuli2 12ny1 13ny2 14xiayouZQ 15diaoduZQ 16pp1 17fd
    cstemp = mssql.ExecQuery("select data from inikuqun where code='%s'" % id)
    cstemp = np.array(cstemp)
    cs = np.zeros(14)
    cs[0] = 0.1  # 目前缩放系数都设置为0.1
    try:
        cs[1] = cstemp[15]  # pp1 发电流量
    except:
        cs[1]=0
    try:
        cs[2] = cstemp[16]  # pp1 发电流量
    except:
        print(id)
        cs[2] = 0

    cs[3] = cstemp[0]  # nn
    cs[4] = cstemp[1]  # xe
    cs[5] = cstemp[2]  # kebuhsi
    cs[6] = cstemp[6]  # qm2
    cs[7] = cstemp[7]  # nnum2
    cs[8] = cstemp[8]  # jizu2
    cs[9] = cstemp[9]  # chuli1
    cs[10] = cstemp[10]  # chuli2
    cs[11] = cstemp[3]  # qm1
    cs[12] = cstemp[4]  # nnum1
    cs[13] = cstemp[5]  # jizu1


    return cs
    # return tez, zqtemp, ZMKD
    # return tez, cs, ZMKD
# 通用函数
import datetime


# def readguize(uuid):
#     # 转换ID 通过数据库读取
#     mssql = get_sql()
#     result = mssql.ExecQuery(
#         "SELECT ID,SYSTEM FROM uuid where RES_CODE='%s' and SYSTEM <>'五座大型水库' and SYSTEM <>'五座大型水库预报'" % uuid)
#     for (row, row2) in result:
#         id = row
#         SYSTEM = row2
#     dbname = 'Principle'
#     # 根据principle_t获取梅汛时间 否则直接读取principle表
#     result = mssql.ExecQuery(
#         "SELECT S_TIME,E_TIME FROM principle_t where RES_CODE='%s'" % (uuid))
#     a = result[0]
#     try:
#         stime = a[0].split('.')
#         etime = a[1].split('.')
#         if datetime.datetime.now().month >= int(stime[0]) and datetime.datetime.now().day >= int(
#                 stime[1]) and datetime.datetime.now().month <= int(etime[0]) and datetime.datetime.now().day <= int(
#             etime[1]):
#             dbname = 'Principle_mx'
#     except:
#         pass
#
#     resList1 = mssql.ExecQuery(
#         "SELECT Q1,Q2,moshi FROM %s where KuNum=%s and SYSTEM='%s' order by Code" % (dbname, id, SYSTEM))
#     array = np.array(resList1)
#     guize = {}
#     guize["Z"] = [round(i, 2) for i in array[:, 0]]
#     guize["CK"] = [Reseroperation.TrimNum(i) for i in array[:, 1]]
#     if array[0, 2] == 1:
#         guize["ZorQ"] = 'Z'
#     else:
#         guize["ZorQ"] = 'Q'
#     return guize


def find_connect(i, connect_point):
    kuname_index = []
    for index, item in enumerate(connect_point):
        if item == str(i + 1):
            kuname_index.append(index)
    return kuname_index


def guizheng(data, num1):
    if len(data) > num1:
        data = data[:num1]
    elif len(data) < num1:
        num2 = num1 - len(data)
        temp = np.zeros(num2) + data[-1]
        data = np.hstack((data, temp))
    return data


def guizhengresw(resw, num):
    for key, value in resw.items():
        temp = np.array(value)
        temp = guizheng(temp, num + 1)
        temp = temp.tolist()
        resw[key] = temp
    return resw

def readybtxt():
    file = open("ybdz.txt", 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return lines

# 找下一个指定类型的节点 # type = 1水库 2断面
def findsk(uuid, mssql, type):
    uuid_temp = uuid
    Type = 0
    while (Type != type):
        next_code = mssql.ExecQuery("SELECT next_code FROM kuname WHERE ID = '%s'" % (uuid_temp))
        next_code = np.array(next_code)
        next_code = next_code[0][0]
        Type_temp = mssql.ExecQuery("SELECT Type FROM kuname WHERE ID = '%s'" % (next_code))
        Type_temp = np.array(Type_temp)
        Type_temp = Type_temp[0][0]
        Type = Type_temp
        uuid_temp = next_code
    return uuid_temp
#计算一条支流演算到断面的天然流量
def find_ggfhd(uuid, mssql, fhdid,qy,dtt,num1,mafacans,tianran):
    uuid_temp = uuid
    while (uuid_temp != fhdid):
        next_code = mssql.ExecQuery("SELECT next_code FROM kuname WHERE ID = '%s'" % (uuid_temp))
        next_code = np.array(next_code)
        next_code = next_code[0][0]
        qytemp = Reseroperation.FD_mafa(qy[uuid_temp],dtt,num1,uuid_temp,next_code,mafacans)
        qytemp = Reseroperation.guizheng(qytemp, num1)
        tianran = tianran + qytemp
        uuid_temp = next_code
    return tianran