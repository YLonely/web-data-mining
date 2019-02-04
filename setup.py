from setuptools import setup, find_packages

setup(
    name='ucas_dm',
    version='1.0.0',
    description="ucas_dm is a simple library provides preprocess and some recommend algorithms for UCAS's web data "
                "mining homework project",
    author='YLonely',
    license='MIT',
    author_email='loneybw@gmail.com',
    url='https://github.com/YLonely/web-data-mining',
    packages=find_packages(),
    data_files=[('ucas_dm/preprocess/stop_words',
                 ['ucas_dm/preprocess/stop_words/stop.txt'])],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only'
    ],
    install_requires=['pandas',
                      'numpy',
                      'gensim',
                      'jieba',
                      'scikit-surprise']
)
