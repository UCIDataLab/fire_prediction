import ftplib
from ftplib import FTP
from time import time
from StringIO import StringIO
import os
import multiprocessing
import logging
from Queue import Queue
import atexit

class Writer(object):
    def __init__(self, write_queue):
        self.write_queue = write_queue

    def write(self):
        while True:
            dest_path, data = self.write_queue.get()
            if data is None:
                logging.debug('Writing queue has reached terminal element, exiting')
                return
            logging.debug('Writing %s, %d items in queue' % (dest_path, self.write_queue.qsize()))
            with open(dest_path, 'wb') as f:
                f.write(data.getvalue())
            data.close()


class AsyncFTPFetcher(object):
    def __init__(self, connection_info, write_queue, blk_size=81920):
        self.connection_info = connection_info
        self.write_queue = write_queue
        self.blk_size = blk_size

        self.setup_ftp()

    def fetch(self, src_path, dest_path):
        out_str = StringIO()

        logging.debug('Fetching %s' % src_path)

        command = 'RETR %s' % src_path

        try:
            self.ftp.retrbinary(command, out_str.write, blocksize=self.blk_size)
        except ftplib.error_temp as err:
            # Reset connection and retry
            logging.debug('Resetting connection')
            self.ftp.close()
            self.setup_ftp()

            self.ftp.retrbinary(command, out_str.write, blocksize=self.blk_size)

        self.write_queue.put((dest_path, out_str))

    def setup_ftp(self):
        self.ftp = FTP(**self.connection_info)


def fetch_thread(context, src_path, dest_path):
    async_fetcher = context['fetcher']
    async_fetcher.fetch(src_path, dest_path)

class PoolLimitedTaskQueue(object):
    def __init__(self, pool_size, task_queue_size, initializer=None, destructor=None):
        self.pool_size = pool_size
        self.initializer = initializer
        self.destructor = destructor

        self.task_queue = multiprocessing.Queue(task_queue_size)
        self.workers = self.create_workers(pool_size)

        self.start_workers()

    def create_workers(self, num_workers):
        workers = []
        for i in range(num_workers):
            context = {}
            w = multiprocessing.Process(target=lambda: self.run(context))
            atexit.register(w.terminate)
            workers.append(w)

        return workers

    def start_workers(self):
        [w.start() for w in self.workers]

    def run(self, context):
        if self.initializer: self.initializer(context)

        while True:
            func, args = self.task_queue.get()
            if func is None:
                logging.debug('Pool worker (%d) stopping after reaching terminal element in task queue' % os.getpid())
                if self.destructor: self.destructor(context)
                return
            apply(func, (context,) + args)

    def apply_async(self, func, args):
        self.task_queue.put((func, args)) 

    def close(self):
        # Put one terminal element for each worker
        for i in range(self.pool_size):
            self.task_queue.put((None, None))

    def terminate(self):
        [w.terminate() for w in self.workers]

    def join(self):
        [w.join() for w in self.workers]

class AsyncFTP(object):
    def __init__(self, server_name, username, password, pool_size, queue_size):
        self.server_name = server_name
        self.username = username
        self.password = password
        self.pool_size = pool_size
        self.queue_size = queue_size

        self.started = False
    
    def start(self):
        self.write_queue = multiprocessing.Queue(maxsize=self.queue_size) # Files to be written

        # Must come after initializing self.queue and self.ftp_queue because self.init_worker() uses them
        self.pool = PoolLimitedTaskQueue(self.pool_size, self.pool_size*2, initializer=self.init_worker, destructor=self.destruct_worker) 

        # Async writing files to disk
        self.writer = multiprocessing.Process(target=Writer(self.write_queue).write)
        self.writer.start()

        self.started = True

    def fetch(self, src_path, dest_path):
        if not self.started:
            raise Exception('Not started')

        self.pool.apply_async(fetch_thread, args=(src_path, dest_path))

    def join(self):
        if not self.started:
            raise Exception('Not started')

        self.pool.close()
        self.pool.join()

        self.write_queue.put((None, None))
        logging.debug('Waiting on writer to finish processing queue')
        self.writer.join()

    def init_worker(self, context):
        connection_info = {'host': self.server_name, 'user': self.username, 'passwd': self.password}
        async_fetcher = AsyncFTPFetcher(connection_info, self.write_queue)
        context['fetcher'] = async_fetcher

        logging.debug('Initialized worker (%d)' % os.getpid())


    def destruct_worker(self, context):
        async_fetcher = context['fetcher']
        async_fetcher.ftp.quit()

        logging.debug('Destructed worker (%d)' % os.getpid())




