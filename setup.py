from setuptools import setup, find_packages

config = {
    'description': 'Keras implementation of 2015 DeepSEA',
    'download_url': 'https://github.com/zj-zhang/deepsea_keras',
    'version': '0.1.0',
    'packages': find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    'setup_requires': [],
    'dependency_links': [],
    'name': 'deepsea_keras',
}

if __name__== '__main__':
    setup(**config)
