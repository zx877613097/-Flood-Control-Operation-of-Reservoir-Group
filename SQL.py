# -*- coding: utf-8 -*-
# import pymssql
from DBUtils.PooledDB import PooledDB
import pymysql
#from PyQt5.QtWidgets import QMessageBox,QDialog
class MSSQL():
    def __init__(self,host,port,user,pwd,db):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db
        self.port = port

    def __GetConnect(self):
        """
        得到连接信息
        返回: conn.cursor()
        """
        if not self.db:
            raise(NameError,"没有设置数据库信息")
        # self.conn = pymysql.connect(host=self.host, port=self.port,user=self.user,password=self.pwd,database=self.db,charset="utf8")#cp936
        #使用连接池
        pool = PooledDB(creator=pymysql,#数据库类型
                        blocking=True,#连接池中如果没有可用连接后是否等待，True等待，False不等待并报错
                        maxconnections=300,#连接池允许的最大连接数
                        mincached=10,#初始化时连接池中至少创建的空闲连接个数，0表示不创建
                        maxcached=300,#最大空闲连接个数，0表示无限制
                        host=self.host, port=self.port,user=self.user,password=self.pwd,database=self.db,charset="utf8")
        self.conn = pool.connection()
        cur = self.conn.cursor()
        if not cur:
            raise(NameError,"连接数据库失败")
        else:
            return cur

    def ExecQuery(self,sql):
        """
        执行查询语句
        返回的是一个包含tuple的list，list的元素是记录行，tuple的元素是每行记录的字段
        resList = ms.ExecQuery("SELECT id,NickName FROM WeiBoUser")
        """
        resList=[]
        try:
            cur = self.__GetConnect()
        except:
            return resList# 查询不上返回空
        cur.execute(sql)
        resList = cur.fetchall()
        self.conn.close()
        return resList

    def ExecNonQuery(self,sql):
        """
        执行非查询语句
            cur = self.__GetConnect()
            cur.execute(sql)
            self.conn.commit()
            self.conn.close()
        """
        try:
            cur = self.__GetConnect()
        except pymssql.OperationalError:
            pass
        cur.execute(sql)
        self.conn.commit()
        self.conn.close()

# conn = pymysql.connect(user='root', password='123456', database='travel', charset='utf8')
# cursor = conn.cursor()
# query = ('select * from tab_category')
# cursor.execute(query)
# for (a) in cursor:
#     print(a)
# cursor.close()
# conn.close()

def main():
    # ms = MSSQL(host="127.0.0.1", port="3306", user="root", pwd="123456", db="anshaKH1")
    # # sql = "SELECT * FROM persons"
    # resList = ms.ExecQuery("SELECT canshu FROM canshu where id =1")  # 查
    ms = MSSQL(host="127.0.0.1", port="3306", user="root", pwd="123456", db="qtj_forecasting_resfloodoperation")
    resList = ms.ExecQuery("SELECT X,Cv,Cs FROM ini_piii where KuNum=%s and NAME='hongfengliuliang'" % 1)
    print(resList)

if __name__ == '__main__':
    main()
