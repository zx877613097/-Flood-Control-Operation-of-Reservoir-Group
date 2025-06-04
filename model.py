import datetime
import json
import math
import traceback
import numpy as np
import copy
from flask import Blueprint, jsonify, request
from Reseroperation import chazhi,dys
import Reseroperation
import publicFA
from log import Logging
from publicFA import get_name, PreRead
from PSO_EP import pso
import os
import pandas as pd
logDetail = Logging()
model = Blueprint('model', __name__)

# 全局变量存储结果
global_resw = {}
global_fhdw = {}
global_zb = {}

# 等效投影分解协调——————————————————————————————————————————————————————
@model.route('/EP', methods=['GET', 'POST'])
def EP():
    try:
        if request.method == 'POST':
            val = request.get_json(force=True)
        else:
            val = request.args.get('val')
            val = json.loads(val)
        resw = calcEP(val)
        # bool = Qchazhi.resw_to_excel(resw, '水位控制模型')
        return jsonify(resw)
    except Exception as e:
        logDetail.UpdateError(str(e), traceback.format_exc())
        return {'errno': 1, 'desc': str(e)}

def calcEP(value):
    '''1  整理Json数据，包含水库的控制条件和各节点的预报来水'''
    qy = value["qy"]  # 预报来水
    tianran = value["tianran"]  # 节点天然来水
    sw = value["sw"]  # 水库最高水位控值
    ck = value["ck"]  # 水库最大出库控值
    ze = value["ze"]  # 水库期末水位约束
    num1 = value["num1"]  # 时段数
    moshi = value["moshi"]  # 水库调度模式
    zmin = value["zmin"]  # 水库最低水位约束
    dq = value["dq"]  # 出库变幅约束
    dtt = value["dtt"]  # 时段长
    q0 = value["q0"]  # 水库当前出库
    z0 = value["z0"]  # 水库当前水位
    lastTime = value["lastTime"]
    method = value["method"]  # 河道演进方法
    systemid = value["systemid"]  # 数据库名称

    # 把单个时间转换成时间序列
    time = value["time"]  # "time":"2017-06-24 00:00:00",
    Time = []
    t = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    for i in range(num1 + 1):
        Time.append(t.strftime("%Y-%m-%d %H"))  # 转换为字符串
        t = t + datetime.timedelta(hours=dtt)  # datetime

    '''2  从数据库取出模型计算需要的数据'''
    nameSK, nameFHD = get_name(systemid)
    mssql = publicFA.get_sql_unique(systemid)  # 连接数据库

    tez = []  # 存放水库水位库容曲线和泄流能力曲线
    cs = []  # 存放水库发电需要数据
    zq = []  # 存放断面水位流量关系曲线

    # 获取每个水库的水位-库容、水位-泄流能力曲线
    result = mssql.ExecQuery("SELECT RES_CODE FROM uuid order by ID")  # 全部读出来后再选
    idlist = np.array(result)
    for i in range(1, len(nameSK) + 1):
        uuid = idlist[i - 1][0]
        temp, cstemp, zqtemp, ZMKD = PreRead(uuid, mssql)
        cstemp[0] = 0
        cs.append({uuid: cstemp})
        tez.append({uuid: temp})
        zq.append({uuid: zqtemp})

    # 获取断面的水位-流量曲线
    FHD_ID = mssql.ExecQuery("SELECT ID FROM Kuname where type='2' order by ID")
    fhd_idlist = np.array(FHD_ID)
    for i in range(len(nameSK) + 1, len(nameSK) + len(nameFHD) + 1):
        result = mssql.ExecQuery(
            "SELECT M_Z,M_DATA FROM fhd_charadata where CH_ID = 10007 and FHD_ID = '%s' order by M_Z" % (
                fhd_idlist[i - len(nameSK) - 1][0]))  # 全部读出来后再选
        zqtemp = np.array(result)
        zqtemp = {fhd_idlist[i - len(nameSK) - 1][0]: zqtemp}
        zq.append(zqtemp)

    # list转字典
    cs = {k: v for d in cs for k, v in d.items()}
    tez = {k: v for d in tez for k, v in d.items()}
    zq = {k: v for d in zq for k, v in d.items()}

    # 读取水库群拓扑关系
    result = mssql.ExecQuery(
        "SELECT ID,next_code,TYPE,sx FROM kuname order by sx" )  # 全部读出来后再选
    kuqun_topo = np.array(result)

    #执行优化

    for i in range(1,6):
        w1 = 0.7+0.01 * i  # 水库权重
        w2 = 1 - w1  # 断面权重

        # 根据权重创建对应的文件夹
        weight_folder = f'weight_{w1:.2f}_{w2:.2f}'
        if not os.path.exists(weight_folder):
            os.makedirs(weight_folder)

        # 创建pso_timing_results子文件夹
        timing_folder = os.path.join(weight_folder, "pso_timing_results")
        if not os.path.exists(timing_folder):
            os.makedirs(timing_folder)

        # 内循环为每个权重方案计算100次
        for j in range(10):
            resw, fhdw, zb, iteration_times, total_time = optimize_with_pso(
                qy, sw, ck, num1, tez, zmin, dq, dtt, ze, q0, z0, moshi, method,
                zq, Time, kuqun_topo, tianran, mssql, w1, w2
            )

            maxQQ = max(fhdw["40105150"]["all"])

            # 创建zb子文件夹
            zb_folder = os.path.join(weight_folder, 'zb')
            if not os.path.exists(zb_folder):
                os.makedirs(zb_folder)

            # 保存zb到Excel
            zb_df = pd.DataFrame.from_dict(zb, orient='index')
            zb_df.to_excel(f'{zb_folder}/zb_{j}_{maxQQ}.xlsx', index=True)

            # 保存resw和fhdw结果到对应权重文件夹
            save_results_to_excel(resw, fhdw, iteration=j, folder=weight_folder)

            # # 保存耗时数据到对应权重文件夹的 pso_timing_results 子文件夹
            # save_timing_to_excel(j, iteration_times, total_time, folder=timing_folder)
# 各子系统的等效投影约束
def reservoir_flood_control_EP(qc, qy, zm, num1, zmin, dq,dtt,ze, q0, z0,tez, qujian,zqtemp):
    from scipy.signal import savgol_filter
    # 使用Savitzky-Golay滤波对水库出库作滑动平均处理
    qc = savgol_filter(qc, window_length=15, polyorder=2)
    # 使用Savitzky-Golay滤波对水库出库作滑动平均处理
    qc = savgol_filter(qc, window_length=15, polyorder=2)
    # qc = savgol_filter(qc, window_length=15, polyorder=2)
    qc[0] =q0
    z1 = tez[0]
    v = tez[1]
    # z2 = tez[2]
    # hs = tez[3]
    z3 = tez[2]
    qxl = tez[3]

    qy = np.insert(qy, 0, qy[0])
    zm = np.insert(zm, 0, zm[0])  # 前端传进来均为0-23 加后变成0-24 实际参与计算的为1-25    ckOrigin = 0，
    qc = np.insert(qc, 0, qc[0])
    qujian = np.insert(qujian, 0, qujian[0])
    zz = np.zeros(num1 + 2)
    vj = np.zeros(num1 + 2)

    mm1 = len(z1)
    mm3 = len(z3)
    # ————————————————————————————————————————————————————————

    # qc[0] = q0
    zz[1] = z0
    vj[1] = chazhi(z1, v, mm1, z0)
    vl = chazhi(z1, v, mm1, zmin)

    qymax = np.max(qy)  # 不超过最大来水的80% （甲级方案确定性系数0.8预报误差） 暂时去掉0.8
    pointh = np.argmax(qy)

    # 基于等效投影的出库流量计算与优化
    for i in range(1, num1 + 1):
        qc[i] = dys(qc[i], qc[i-1], dq) #变幅约束
        dd = chazhi(z3, qxl, mm3, zz[i])#泄流能力约束
        if qc[i] > dd:
            qc[i] = dd

        if qc[i] < 0:
            qc[i] = 0
        if qc[i] > qymax:
            qc[i] = qymax

        # 水量平衡等效约束
        sum_flow_diff = np.sum([(qy[m] - qc[m]) * dtt for m in range(2, i)])
        vj[i + 1] = vj[1] + (sum_flow_diff / 1000000 * dtt * 3600) + (
                    (qy[1] - qc[1]) + (qy[i] - qc[i])) / 2000000 * dtt * 3600
        #vj[i + 1] = vj[i] + (qy[i] + qy[i - 1] - qc[i] - qc[i - 1]) / 2000000 * dtt * 3600

        zz[i + 1] = chazhi(v, z1, mm1, vj[i + 1])

    # 水库蓄水量上下限等效约束
    zmmax = np.max(zz)
    pointh = np.argmax(zz)  # 第一个最高水位点
    kkk = 0
    while (kkk < 10 and zmmax- zm[num1 - 5]>0):
        dv1 = (chazhi(z1, v, mm1,zmmax) - chazhi(z1, v, mm1, zm[num1 - 5])) / dtt / 3600 * 1000000
        dv2 = 0
        kk = 1  # 从pointl点处起调
        for i in range(kk + 1, pointh + 1):
            # ① 计算最高水位低于zm，最高水位以前少放水
            # ② 计算最高水位高于zm，若之前达到过死水位，则从最低点到最高点多放水。若未达到过死水位，最高水位以前多放水。
            dv2 = dv2 + qc[i - 1]
        if dv2 != 0:
            af = 1 + dv1 / dv2
        else:
            af = 1
        for i in range(kk + 1, num1 + 1):
            # for i in range(kk + 1, pointh + 1):
            if af != 1:  # benmx
                qc[i - 1] = af * qc[i - 1]
            # else:
            #     qc[i - 1] = dv1 / (pointh + 1 - kk)  # benmx
        blnQxl = 0  # 2020.4.23 yao
        ####################################
        ######################################
        for i in range(kk, num1 + 1):
            if i == 1:
                # qc[0] = qc[0] + pq
                dd = chazhi(z3, qxl, mm3, z0)
                if qc[0] > dd:
                    qc[0] = dd
                if abs(qc[0] - q0) > dq:
                    if qc[0] > q0:
                        qc[0] = q0 + dq
                    if qc[0] < q0:
                        qc[0] = q0 - dq
                if qc[0] < 0:
                    qc[0] = 0
                if qc[0] > qymax:
                    qc[0] = qymax
            if abs(qc[i] - qc[i - 1]) > dq:
                if qc[i] > qc[i - 1]:
                    qc[i] = qc[i - 1] + dq
                if qc[i] < qc[i - 1]:
                    qc[i] = qc[i - 1] - dq
            dd = chazhi(z3, qxl, mm3, zz[i])
            if qc[i] > dd:
                qc[i] = dd
                blnQxl = 1  # 2020.4.23 yao 此时被约束了 需要进入floodCal
            if qc[i] < 0:
                qc[i] = 0
            if qc[i] > qymax:
                qc[i] = qymax
            # 水量平衡等效约束
            sum_flow_diff = np.sum([(qy[m] - qc[m]) * dtt for m in range(2, i)])
            vj[i + 1] = vj[1] + (sum_flow_diff / 1000000 * dtt * 3600) + (
                        (qy[1] - qc[1]) + (qy[i] - qc[i])) / 2000000 * dtt * 3600
            if vj[i + 1] < vl:
                if qc[i] >= qy[i]:
                    qc[i] = qy[i]
                blnQxl = 0
                dd = chazhi(z3, qxl, mm3, zz[i])
                if qc[i] > dd:
                    qc[i] = dd
                    blnQxl = 1
                if abs(qc[i] - qc[i - 1]) > dq:
                    if qc[i] > qc[i - 1]:
                        qc[i] = qc[i - 1] + dq
                    if qc[i] < qc[i - 1]:
                        qc[i] = qc[i - 1] - dq
                # 水量平衡等效约束
                sum_flow_diff = np.sum([(qy[m] - qc[m]) * dtt for m in range(2, i)])
                vj[i + 1] = vj[1] + (sum_flow_diff / 1000000 * dtt * 3600) + (
                            (qy[1] - qc[1]) + (qy[i] - qc[i])) / 2000000 * dtt * 3600
            zz[i + 1] = chazhi(v, z1, mm1, vj[i + 1])
            # 2020.4.23 yao

        # print(kkk)
        kkk = kkk + 1
        zmmax = np.max(zz)
        pointh = np.argmax(zz)

    k=1
    while k <16:
        if abs(ze - zz[num1 + 1]) > 0.005:
            dv = chazhi(z1, v, mm1, zz[num1 + 1]) - chazhi(z1, v, mm1, ze)
            dv1 = dv / dtt / 3600 * 1000000
            if pointh < num1:
                pq = dv1 / (num1-pointh)
                blnQxl = 0  # 2020.4.23 yao
                for i in range(pointh+1, num1 + 1):

                    qc[i] = qc[i] + pq
                    if abs(qc[i] - qc[i - 1]) > dq:
                        if qc[i] > qc[i - 1]:
                            qc[i] = qc[i - 1] + dq
                        if qc[i] < qc[i - 1]:
                            qc[i] = qc[i - 1] - dq
                    dd = chazhi(z3, qxl, mm3, zz[i])
                    if qc[i] > dd:
                        qc[i] = dd
                        blnQxl = 1  # 2020.4.23 yao 此时被约束了 需要进入floodCal
                    if qc[i] < 0:
                        qc[i] = 0
                    # 水量平衡等效约束
                    sum_flow_diff = np.sum([(qy[m] - qc[m]) * dtt for m in range(2, i)])
                    vj[i + 1] = vj[1] + (sum_flow_diff / 1000000 * dtt * 3600) + (
                                (qy[1] - qc[1]) + (qy[i] - qc[i])) / 2000000 * dtt * 3600
                    if vj[i + 1] < vl:
                        qc[i] = qy[i]
                        # qc[i] = qy[i - 1]
                        blnQxl = 0
                        dd = chazhi(z3, qxl, mm3, zz[i])
                        if qc[i] > dd:
                            qc[i] = dd
                            blnQxl = 1  # 2020.4.23 yao 此时被约束了 需要进入floodCal
                        # 水量平衡等效约束
                        sum_flow_diff = np.sum([(qy[m] - qc[m]) * dtt for m in range(2, i)])
                        vj[i + 1] = vj[1] + (sum_flow_diff / 1000000 * dtt * 3600) + (
                                    (qy[1] - qc[1]) + (qy[i] - qc[i])) / 2000000 * dtt * 3600
                    zz[i + 1] = chazhi(v, z1, mm1, vj[i + 1])
            else:
                    pq = dv1 / (num1)
                    for i in range(1, num1 + 1):

                        qc[i] = qc[i] + pq
                        if abs(qc[i] - qc[i - 1]) > dq:
                            if qc[i] > qc[i - 1]:
                                qc[i] = qc[i - 1] + dq
                            if qc[i] < qc[i - 1]:
                                qc[i] = qc[i - 1] - dq
                        dd = chazhi(z3, qxl, mm3, zz[i])
                        if qc[i] > dd:
                            qc[i] = dd
                            blnQxl = 1  # 2020.4.23 yao 此时被约束了 需要进入floodCal
                        if qc[i] < 0:
                            qc[i] = 0
                        # 水量平衡等效约束
                        sum_flow_diff = np.sum([(qy[m] - qc[m]) * dtt for m in range(2, i)])
                        vj[i + 1] = vj[1] + (sum_flow_diff / 1000000 * dtt * 3600) + (
                                (qy[1] - qc[1]) + (qy[i] - qc[i])) / 2000000 * dtt * 3600
                        if vj[i + 1] < vl:
                            qc[i] = qy[i]
                            # qc[i] = qy[i - 1]
                            blnQxl = 0
                            dd = chazhi(z3, qxl, mm3, zz[i])
                            if qc[i] > dd:
                                qc[i] = dd
                                blnQxl = 1  # 2020.4.23 yao 此时被约束了 需要进入floodCal
                            # 水量平衡等效约束
                            sum_flow_diff = np.sum([(qy[m] - qc[m]) * dtt for m in range(2, i)])
                            vj[i + 1] = vj[1] + (sum_flow_diff / 1000000 * dtt * 3600) + (
                                    (qy[1] - qc[1]) + (qy[i] - qc[i])) / 2000000 * dtt * 3600
                        zz[i + 1] = chazhi(v, z1, mm1, vj[i + 1])
        k = k+1

    resw = {}
    resw['sw'] = zm
    # resw['ck'] = ztAlways
    resw['qy'] = qy
    resw['zz'] = zz
    resw['qc'] = qc
    resw["vj"] = vj
    # resw["pp1"] = pp1
    resw["qujian"] = qujian
    resw["num1"] = num1
    resw["zqtemp"] = zqtemp
    resw["dtt"] = dtt
    resw["xy"] = qc+qujian
    return resw

#上层系统目标函数/
def upper_level_objective_parallel(qc, qy, sw, ck, num1, tez, zmin, dq, dtt, ze, q0, z0, moshi, method, zq, Time, kuqun_topo, tianran, mssql,w1,w2):
    global global_resw, global_fhdw, global_zb

    reqy = copy.deepcopy(qy)
    skxy = {}
    fhd = {}

    # 计算各个水库的风险度和其他指标（小浪底、陆浑、故县、河口村）
    ids = ["40104690", "41701500","41602500", "41605400"]
    xishu =[0.739838170737514,0.124617946795438,0.0469813525230241,0.0885625299440245]
    S_total = 0
    for idx, id in enumerate(ids):
        xyid = "40105150"
        Temp = reservoir_flood_control_EP(qc[idx * num1:(idx + 1) * num1], qy[id], sw[id], int(num1), zmin[id], dq[id], dtt, ze[id], q0[id], z0[id], tez[id], reqy[xyid], zq[id])
        reswTemp = Reseroperation.sktez(Temp, id, xyid, method[id], mssql)
        global_resw[id] = reswTemp
        reqy[xyid] = np.array(global_resw[id]["qcMsjg"][1:]) + reqy[xyid]
        zbtemp = Reseroperation.ZhiBiaoCalu(global_resw[id], zq[id], Time)

        skxyTemp = Reseroperation.xytez(global_resw[id]['qcMsjg'], reqy[xyid], zq[xyid], num1)
        skxy[id] = skxyTemp

        # 计算水库的风险度
        z1 = tez[id][0]
        v = tez[id][1]
        mm1 = len(z1)
        v0 = chazhi(z1, v, mm1, z0[id]) #单位是亿立方米
        v_control = chazhi(z1, v, mm1, sw[id][0])  # 使用最高水位进行计算 #单位是百万立方米
        S = (max(global_resw[id]["vj"])-v0) / (v_control - v0) * xishu[idx]
        zbtemp["目标值"] = S
        global_zb[id] = zbtemp
        mubiaozhi = {}
        # print("---")
        # print(max(global_resw[id]["vj"]))
        # print((v_control - v0))
        S_total += S

    # 防洪点的风险度计算
    id = "40105150"
    qy_all = np.insert(np.array(reqy[id]), 0, np.array(reqy[id])[0])
    fhd["all"] = qy_all.tolist()
    fhd["tianran"] = publicFA.guizheng(tianran[id], num1 + 1).tolist()
    fhd["qujian"] = publicFA.guizheng(qy[id], num1 + 1).tolist()
    zz = zq[id][:, 0]
    qq = zq[id][:, 1]
    fhd_zz = [round(Reseroperation.chazhi(qq, zz, len(zz), q), 2) for q in qy_all]
    fhd['fhdzz'] = fhd_zz
    global_fhdw[id] = fhd
    zbtemp = Reseroperation.ZhiBiaoCalu_fhd(global_fhdw[id], zq[id], Time)


    # 防洪点风险度
    if max(fhd["all"])>7300:
        F2 = math.exp(10*((max(fhd["all"])-7300) / 7300))
    else:
        F2 = max(fhd["all"])/7300
    global_zb[id] = zbtemp
    # 计算目标函数初步值
    total_sum = S_total * w1 + F2 * w2

    # 增加罚函数1 如果超最高水位，惩罚
    penalty1 = 1000
    penalty2 = 100
    k = 10000  # 惩罚系数，可以根据具体情况调整
    # #流量平方和最小
    # list1 = global_resw["40104690"]["qcMsjg"]
    # list2 = global_resw["41602500"]["qcMsjg"]
    # list3 = global_resw["41605400"]["qcMsjg"]
    # list4 = global_resw["41701500"]["qcMsjg"]
    # list5 = qy["40105150"]
    # # list5 = list5_temp.insert(0,list5_temp[0])
    # # 逐列求和
    # column_sums = [sum(x) for x in zip(list1, list2, list3, list4,list5)]
    # # 平方和
    # result = math.sqrt(sum(x ** 2 for x in column_sums))
    # result=result/62215
    # print(result)
    # total_sum = result+ total_sum


    # # 最终目标函数值
    # print(total_sum)
    return total_sum

# 为每个水库出库流量生成不同的上下界
def generate_bounds(id, num1,zmin_dict,sw_dict):
    lb = zmin_dict[id] * num1  # 每个水库的最小出库流量
    ub = sw_dict[id] # 每个水库的最大出库流量根据其最大出库控值确定
    return lb, ub

def optimize_with_pso(qy_dict, sw_dict, ck_dict, num1, tez, zmin_dict, dq_dict, dtt, ze_dict, q0_dict, z0_dict, moshi, method, zq, Time, kuqun_topo, tianran, mssql,w1,w2):

    total_sum = 0
    global global_resw, global_fhdw, global_zb


    # 初始化全局变量
    global_resw = {}
    global_fhdw = {}
    global_zb = {}

    # 定义每个水库 168 小时的水位上下界（最小和最大水位）
    lb = [0] * num1 + [0] * num1+ [0] * num1 + [0] * num1  # 每个水库的最小水位不同
    ub = [6000] * num1 + [3000] * num1 + [3000] * num1 + [3000] * num1 # 每个水库的最大水位不同


    args = (
    qy_dict, sw_dict, ck_dict, num1, tez, zmin_dict, dq_dict, dtt, ze_dict, q0_dict, z0_dict, moshi, method, zq,
    Time, kuqun_topo, tianran, mssql,w1,w2)

    best_qc, total_sum,iteration_times,total_time = pso(upper_level_objective_parallel, lb, ub, args=args)

    return global_resw, global_fhdw, global_zb, iteration_times,total_time


def save_timing_to_excel(scheme_id, iteration_times, total_time, folder="pso_timing_results"):
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 定义文件路径和文件名
    filename = os.path.join(folder, "pso_timing_results.xlsx")

    # 创建 DataFrame 记录迭代耗时和总耗时
    df = pd.DataFrame({
        "Iteration": list(range(1, len(iteration_times) + 1)),
        "Iteration_Time (s)": iteration_times
    })
    df.loc[len(df)] = ["Total", total_time]  # 添加总耗时

    # 检查 Excel 文件是否存在，如果不存在则创建新文件
    if not os.path.isfile(filename):
        # 直接保存为新 Excel 文件
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f'Scheme_{scheme_id}', index=False)
    else:
        # 如果文件已存在，则以追加模式添加新工作表
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=f'Scheme_{scheme_id}', index=False)

    print(f'Timing results for Scheme {scheme_id} saved to {filename}')
def save_results_to_excel(resw, fhdw, iteration, folder="results"):
    """
    将 resw 和 fhdw 的结果保存到一个 Excel 文件中。

    :param resw: 水库优化结果字典
    :param fhdw: 防洪点数据字典
    :param iteration: 当前的迭代次数，用于命名列
    :param folder: 保存文件的文件夹
    """
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 定义文件名
    filename = f"result_iteration_{iteration}.xlsx"
    filepath = os.path.join(folder, filename)

    # 使用 Excel writer 对象
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # 保存 resw 数据
        for reservoir_id, res_data in resw.items():
            df_res_data = pd.DataFrame.from_dict(res_data)
            # 保存到 Excel 中的一个 Sheet，Sheet 名称为水库 ID
            sheet_name = f"Reservoir_{reservoir_id}"
            df_res_data.to_excel(writer, sheet_name=sheet_name, index=False)

        # 保存 fhdw 数据
        for fhd_id, fhd_data in fhdw.items():
            if "all" in fhd_data:
                fhd_all_data = pd.DataFrame(fhd_data["all"], columns=[f"Iteration_{iteration}"])
                sheet_name = f"FHD_{fhd_id}_all"
                fhd_all_data.to_excel(writer, sheet_name=sheet_name)

    print(f"Results saved to {filepath}")