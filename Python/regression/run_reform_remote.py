import argparse
import os

import requests

JENKINS_URL = os.environ.get("JENKINS_URL")
if not JENKINS_URL:
    raise ValueError('Define JENKINS_URL env variable to continue, e.g. "http://54.159.16.16:8080/"')
if JENKINS_URL.endswith('/'):
    JENKINS_URL = JENKINS_URL[:-1]

JENKINS_TOKEN = os.environ.get('JENKINS_TOKEN')
if not JENKINS_TOKEN:
    raise ValueError("Expected env var JENKINS_TOKEN to be defined.  See the the Jenkins job config for token already created.")

def url_for_args(**kwargs):
    kwargs['cause'] = 'change'
    kwargs['token'] = JENKINS_TOKEN
    kws = ['{}={}'.format(k, v) for k,v in kwargs.items()]
    kws = '&'.join(kws)
    return '{}/buildWithParameters?{}'.format(JENKINS_URL, kws)


def cli():
    parser = argparse.ArgumentParser(description="Submit regression tests to Jenkins")
    parser.add_argument('reformid', help='Reform id such as "reform0" "reform9"')
    parser.add_argument('--install_taxcalc_version', help='Tax-Calc version to install', default='0.6.6')
    parser.add_argument('--install_ogusa_version', default='master', help='OG-USA version to install')
    parser.add_argument('--diff', action='store_true', help='Run difference against standards?')
    parser.add_argument('--compare_taxcalc_version', help='Version of TaxCalc for comparison', default='0.6.6')
    parser.add_argument('--compare_ogusa_version', help='Version of OG-USA to compare against', default='0.5.5')
    parser.add_argument('--ogusainstallmethod', default='git', choices=['conda', 'git'], help='Enter "conda" or "git" If it is "git" then installogusa version will be checked out ')
    return parser.parse_args()

def submit():
    args = cli()
    url = url_for_args(**vars(args))
    print('GET', url)
    response = requests.post(url)
    if response.status_code != 200:
        raise ValueError("Bad response code {} \n\n{}".format(response.status_code, response._content))

if __name__ == "__main__":
    submit()