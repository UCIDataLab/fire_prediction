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
                logging.debug('Writing queue has reached terminal element, exiting')
                return
            logging.debug('Writing %s, %d items in queue' % (dest_path, queue.qsize()))
            with open(dest_path, 'wb') as f:
                f.write(data.getvalue())
            data.close()
            self.queue.task_done()

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

class PoolLimitedTaskQueue(multiprocessing.Pool):
    def __init__(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None, context=None, taskqueue_maxsize=None):
        self._ctx = context or get_context()
        self._setup_queues()
        self._taskqueue = queue.Queue(maxsize=taskqueue_maxsize)
        self._cache = {}
        self._state = RUN
        self._maxtasksperchild = maxtasksperchild
        self._initializer = initializer
        self._initargs = initargs

        if processes is None:
            processes = os.cpu_count() or 1
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")

        if initializer is not None and not callable(initializer):
            raise TypeError('initializer must be a callable')

        self._processes = processes
        self._pool = []
        self._repopulate_pool()

        self._worker_handler = threading.Thread(
            target=Pool._handle_workers,
            args=(self, )
            )
        self._worker_handler.daemon = True
        self._worker_handler._state = RUN
        self._worker_handler.start()


        self._task_handler = threading.Thread(
            target=Pool._handle_tasks,
            args=(self._taskqueue, self._quick_put, self._outqueue,
                  self._pool, self._cache)
            )
        self._task_handler.daemon = True
        self._task_handler._state = RUN
        self._task_handler.start()

        self._result_handler = threading.Thread(
            target=Pool._handle_results,
            args=(self._outqueue, self._quick_get, self._cache)
            )
        self._result_handler.daemon = True
        self._result_handler._state = RUN
        self._result_handler.start()

        self._terminate = util.Finalize(
            self, self._terminate_pool,
            args=(self._taskqueue, self._inqueue, self._outqueue, self._pool,
                  self._worker_handler, self._task_handler,
                  self._result_handler, self._cache),
            exitpriority=15
            )

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
        self.pool = PoolLimitedTaskQueue(self.pool_size, initializer=self.init_worker, taskqueue_maxsize=10) 

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

        self.queue.put((None, None))
        logging.debug('Waiting on writer queue')
        self.queue.join()
        logging.debug('Waiting on writer to finish processing queue')
        self.writer.join()

    def init_worker(self):
        ftp = FTP(self.server_name)
        ftp.login(self.username, self.password)

        global async_fetcher
        async_fetcher = AsyncFTPFetcher(ftp, self.queue)

        logging.debug('Initialized worker')




