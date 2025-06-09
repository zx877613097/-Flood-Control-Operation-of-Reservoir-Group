# -*- coding: utf-8 -*-
import os
from flask import Flask

from log import Logging
logDetail = Logging()

from model import model
# from model2 import model2

app = Flask(__name__)
# manager = Manager(app)
# 注册蓝图，并指定其对应的前缀（url_prefix）
app.register_blueprint(model)
# app.register_blueprint(model2)

import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(level=logging.ERROR)  # 日志等级为ERROR
fh = RotatingFileHandler(filename="log/myapp.log", maxBytes=1024 * 1024 * 100, backupCount=100)
formatter = logging.Formatter(
    '[%(asctime)s] - %(filename)s [Line:%(lineno)d] - [%(levelname)s]-[thread:%(thread)s]-[process:%(process)s] - %(message)s')
fh.setFormatter(formatter)
logging.getLogger().addHandler(fh)
os.putenv("FLASK_ENV", "development")



# 以上频率曲线相关—————————————————————————————————————————————————————

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)

    # http_server = WSGIServer(('0.0.0.0', 24625), app, handler_class=WebSocketHandler)
    # print("Server started")
    # http_server.serve_forever()

    # manager.run()
