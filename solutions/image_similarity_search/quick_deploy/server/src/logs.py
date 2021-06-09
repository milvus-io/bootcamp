import os
import re
import datetime
import logging
import sys

try:
    import codecs
except ImportError:
    codecs = None

from pymilvusdm.setting import WORK_PATH, LOGS_NUM


# LOGS_NUM = os.getenv("logs_num", 0)
# # WORK_PATH = '/Users/bennu/workspace/git/milvus_data_migration/pymilvusdm/core'
# WORK_PATH = os.path.join(os.environ['HOME'], 'milvusdm')

class MultiprocessHandler(logging.FileHandler):
    def __init__(self, filename, when='D', backupCount=0, encoding=None, delay=False):
        self.prefix = filename
        self.backupCount = backupCount
        self.when = when.upper()
        self.extMath = r"^\d{4}-\d{2}-\d{2}"

        self.when_dict = {
            'S': "%Y-%m-%d-%H-%M-%S",
            'M': "%Y-%m-%d-%H-%M",
            'H': "%Y-%m-%d-%H",
            'D': "%Y-%m-%d"
        }

        self.suffix = self.when_dict.get(when)
        if not self.suffix:
            print('The specified date interval unit is invalid: ', self.when)
            sys.exit(1)
            # raise ValueError(u"指定的日期间隔单位无效: %s" % self.when)

        self.filefmt = os.path.join(WORK_PATH, "logs", "%s-%s.log" % (self.prefix, self.suffix))
        # print('filefmt: ', self.filefmt)

        self.filePath = datetime.datetime.now().strftime(self.filefmt)
        # print('filepath: ', self.filePath)

        _dir = os.path.dirname(self.filefmt)
        # print('logs_dir: ', _dir)
        try:
            # 如果日志文件夹不存在，则创建文件夹
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        except Exception as e:
            print('Failed to create log file: ', e)
            print("log_path：" + self.filePath)
            sys.exit(1)

        if codecs is None:
            encoding = None

        logging.FileHandler.__init__(self, self.filePath, 'a+', encoding, delay)

    def shouldChangeFileToWrite(self):
        _filePath = datetime.datetime.now().strftime(self.filefmt)
        if _filePath != self.filePath:
            self.filePath = _filePath
            return True
        return False

    def doChangeFile(self):
        """输出信息到日志文件，并删除多于保留个数的所有日志文件"""
        # 日志文件的绝对路径
        self.baseFilename = os.path.abspath(self.filePath)
        if self.stream:
            self.stream.close()
            # 关闭stream后必须重新设置stream为None，否则会造成对已关闭文件进行IO操作。
            self.stream = None

        if not self.delay:
            self.stream = self._open()
        if self.backupCount > 0:
            # print('-----------delete backup logs------------')
            for s in self.getFilesToDelete():
                # print(s)
                os.remove(s)

    def getFilesToDelete(self):
        """获得过期需要删除的日志文件"""
        dir_name, _ = os.path.split(self.baseFilename)
        file_names = os.listdir(dir_name)
        result = []
        prefix = self.prefix + '-'
        # plen = len(prefix)
        for file_name in file_names:
            if file_name[:len(prefix)] == prefix:
                suffix = file_name[len(prefix):-4]
                if re.compile(self.extMath).match(suffix):
                    result.append(os.path.join(dir_name, file_name))
        result.sort()

        # 返回  待删除的日志文件
        #   多于 保留文件个数 backupCount的所有前面的日志文件。
        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result

    def emit(self, record):
        """发送一个日志记录
        覆盖FileHandler中的emit方法，logging会自动调用此方法"""
        try:
            if self.shouldChangeFileToWrite():
                self.doChangeFile()
            logging.FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def write_log():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # formatter = '%(asctime)s ｜ %(levelname)s ｜ %(filename)s ｜ %(funcName)s ｜ %(module)s ｜ %(lineno)s ｜ %(message)s'
    fmt = logging.Formatter(
        '%(asctime)s ｜ %(levelname)s ｜ %(filename)s ｜ %(funcName)s ｜ %(lineno)s ｜ %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    log_name = "milvusdm"
    file_handler = MultiprocessHandler(log_name, when='D', backupCount=LOGS_NUM)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler.doChangeFile()

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    message = 'test writing logs'
    logger = write_log()
    logger.info(message)
    logger.debug(message)
    logger.error(message)