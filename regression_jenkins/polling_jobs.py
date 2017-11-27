import argparse
import time
import os
import re

import bs4
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

POLL_INTERVAL = 18 # seconds
OSPC_API_KEY = os.environ['OSPC_API_KEY']
JENKINS_DOMAIN= os.environ["JENKINS_DOMAIN"]

BUILD_CAUSE = os.environ['BUILD_CAUSE']

HASHES = {}

def cli():
    parser = argparse.ArgumentParser(description='Wait - get results')
    parser.add_argument('reforms', nargs='+', help='1 or more reform names')
    return parser.parse_args()


def request(url, requests_method='get', **kwargs):
    print('Request', url)
    kw = dict(auth=HTTPBasicAuth('ospctaxbrain', OSPC_API_KEY), **kwargs)
    req = getattr(requests, requests_method)(url, **kw)
    content = req._content
    if req.status_code != 200:
        raise ValueError(req._content)
    return content

def get_log(reform, build_num='lastBuild'):
    url = '{}/job/ci-mode-simple-{}/{}/consoleText'.format(JENKINS_DOMAIN, reform, build_num)
    return request(url)

def workspace_url(reform):
    return '{}/job/ci-mode-simple-{}/ws/OG-USA/regression/'.format(JENKINS_DOMAIN, reform)

def get_workspace_main(reform):
    return request(workspace_url(reform))

def get_diff_files(reform):
    if not os.path.exists('artifacts'):
        os.mkdir('artifacts')
    content = get_workspace_main(reform)
    for a in bs4.BeautifulSoup(content, 'lxml').find_all('a'):
        href = a.get('href', '')
        if 'txt' in href  or 'csv' in href:
            if '*view*' in href:
                continue
            if href.startswith(('results', 'diff')):
                url = workspace_url(reform) + href
                content = request(url)
                dirr = os.path.join(artifacts, reform)
                if not os.path.exists(dirr):
                    os.mkdir(dirr)
                fname = os.path.join(dirr, href)
                with open(fname, 'w') as f:
                    f.write(content)
                print('Wrote {}'.format(os.path.abspath(fname)))

def is_started_finished(reform, build_num='lastBuild'):
    log = get_log(reform, build_num=build_num)
    lines = filter(None, log.splitlines())
    if lines:
        first, last = lines[0], lines[-1]
    else:
        first, last = '', ''
    started = first.strip().startswith('Started')
    finished = last.strip().startswith('Finished') and len(lines) > 500
    return lines, started, finished


def find_build_number(reform, max_wait=300,
                      build_num='lastBuild',
                      try_once=False):
    content = request('{}/job/ci-mode-simple-{}/lastBuild/'.format(JENKINS_DOMAIN, reform))
    if build_num == 0:
        build_num = 'lastBuild'
    for h1 in bs4.BeautifulSoup(content, 'lxml').find_all('h1'):
        txt = h1.get_text()
        mat = re.search('Build\s+#\s*(\d+)\s+', txt)
        if bool(mat):
            if build_num == 'lastBuild':
                build_num = mat.groups()[0]

            start = time.time()
            retries = 0
            while True:
                try:
                    lines, started, finished = is_started_finished(reform, build_num)
                    if lines and BUILD_CAUSE.lower() in ' '.join(lines).lower():
                        print('Found build number is {}'.format(build_num))
                        break
                    print('Failed to find build_num', build_num)
                    build_num = find_build_number(reform,
                                                  max_wait=max_wait,
                                                  build_num=build_num,
                                                  try_once=True)
                    build_num -= 1

                except:
                    build_num = 'lastBuild'
                    if retries > 100 or try_once:
                        raise
                    retries += 1
                    time.sleep(10)
    return build_num


def get_results():
    global HASHES
    args = cli()
    build_nums = {}
    for reform in args.reforms:
        build_nums[reform] = find_build_number(reform)
    reforms_outstanding = set(args.reforms)
    poll = POLL_INTERVAL / 6
    last_print = time.time()
    while reforms_outstanding:
        for reform in tuple(reforms_outstanding):
            lines, started, finished = is_started_finished(reform, build_num=build_nums[reform])
            if time.time() - last_print > 60 * 2:
                # Avoid build getting killed for lack of IO
                print('Polling...')
                last_print = time.time()
            if not started:
                print("WARNING: Expected started to be True - lines : {}".format('\n'.join(lines)))
            if finished:
                print('REFORM FINISHED: {}'.format(reform))
                reforms_outstanding.remove(reform)
                print('Waiting on {}'.format(reforms_outstanding))
                log_dir = os.path.join('artifacts', reform)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                with open(os.path.join(log_dir, 'console_raw_log.txt'), 'w') as f:
                    f.write('\n'.join(lines))
            else:
                time.sleep(poll)
            poll = (3 * poll + POLL_INTERVAL) / 4 # gradually up to POLL_INTERVAL
    for reform in args.reforms:
        print('Get diff / results files for {}'.format(reform))
        get_diff_files(reform)

    results_data_files = []
    for root, dirs, files in os.walk('artifacts'):
        for f in files:
            f = os.path.abspath(os.path.join(root, f))
            if 'pprint' in f:
                with open(f) as f2:
                    content = f2.read()
                print('From artifact {}:\n\n{}\n\n'.format(f, content))
            if f.startswith('results_data') and f.endswith('.csv'):
                results_data_files.append(f)
    if not os.path.exists('artifacts'):
        os.mkdir('artifacts')
    if results_data_files:
        dataframes = [pd.read_csv(d, index_col=0) for d in results_data_files]
        keys = [os.path.basename(d).replace('results_data_', '').replace('.csv', '') for di in ds]
        df = pd.concat(dataframes, keys=keys)
        df.index.names = 'reform', 'category'
        df.to_csv('artifacts/results_data_concat.csv')
    else:
        with open('artifacts/results_data_concat.csv', 'w') as f:
            f.write('No artifacts')
    print('ok')

if __name__ == '__main__':
    get_results()
