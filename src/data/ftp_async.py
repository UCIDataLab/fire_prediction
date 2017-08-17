from ftplib import FTP
from time import time
from StringIO import StringIO
import os
import multiprocessing
import logging

class Writer(object):
    def __init__(self, queue):
        self.queue = queue

    def write(self):
        while True:
            dest_path, data = self.queue.get()
            if data == None:
                return
            with open(dest_path, 'wb') as f:
                f.write(data.getvalue())
            data.close()

async_fetcher = None

class AsyncFTPFetcher(object):
    def __init__(self, ftp, queue, blk_size=81920):
        self.ftp = ftp
        self.queue = queue
        self.blk_size = blk_size
        pass

    def fetch(self, src_path, dest_path):
        out_str = StringIO()

        logging.debug('Fetching %s' % src_path)

        command = 'RETR %s' % src_path
        self.ftp.retrbinary(command, out_str.write, blocksize=self.blk_size)

        self.queue.put((dest_path, out_str))


def fetch_thread(src_path, dest_path):
    async_fetcher.fetch(src_path, dest_path)

class AsyncFTP(object):
    # TODO: Close opened FTP connections after done with pool

    def __init__(self, server_name, username, password, pool_size, queue_size):
        self.server_name = server_name
        self.username = username
        self.password = password
        self.pool_size = pool_size
        self.queue_size = queue_size

        self.started = False
    
    def start(self):
        self.queue = multiprocessing.Queue(maxsize=self.queue_size) # Files to be written

        # Must come after initializing self.queue and self.ftp_queue because self.init_worker() uses them
        self.pool = multiprocessing.Pool(self.pool_size, initializer=self.init_worker) 

        # Async writing files to disk
        self.writer = multiprocessing.Process(target=Writer(self.queue).write)
        self.writer.start()

        self.started = True

    def fetch(self, src_path, dest_path):
        if not self.started:
            raise Exception('Not started.')

        self.pool.apply_async(fetch_thread, args=(src_path, dest_path))

    def join(self):
        if not self.started:
            raise Exception('Not started.')

        self.pool.close()
        self.pool.join()

        logging.debug('Waiting on writer to finish processing queue')
        self.queue.put((None, None))
        self.writer.join()

    def init_worker(self):
        ftp = FTP(self.server_name)
        ftp.login(self.username, self.password)

        global async_fetcher
        async_fetcher = AsyncFTPFetcher(ftp, self.queue)

        logging.debug('Initialized worker')




