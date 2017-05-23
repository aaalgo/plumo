from distutils.core import setup, Extension

_plumo = Extension('_plumo',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y'], 
        libraries = ['boost_python', 'glog'],
        include_dirs = ['/usr/local/include'],
        library_dirs = ['/usr/local/lib'],
        sources = ['plumo.cpp']
        )

setup (name = 'plumo',
       version = '0.0.1',
       author = 'Wei Dong and Yuanfang Guan',
       author_email = 'wdong@wdong.org',
       license = 'MIT',
       description = 'This is a demo package',
       ext_modules = [_plumo],
       )
