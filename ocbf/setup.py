# setup.py
from setuptools import Extension, setup, find_packages

selection_extension = Extension(
    'ocbf.selection._min_cover_exact',
    sources=['ocbf/selection/_min_cover_exact.cpp'],
    language='c++',
    extra_compile_args=['-O3'],
)

setup(
    name='ocbf',
    version='1.0',
    packages=find_packages(include=['ocbf', 'ocbf.*']),
    package_data={'ocbf': ['mtp_templates/*.mtp', 'default_reduce_assets/*.mtp']},
    author='Jing Huang',
    author_email='2760344463@qq.com',
    description='OCBF active-learning workflow',
    install_requires=[
    ],
    ext_modules=[selection_extension],
    entry_points={
        'console_scripts': [
            'ocbf=ocbf.cli:main',
        ]
    }
)


'''
Remark:
install pymlip
install vaspkit
'ase==3.23.0',
'''
