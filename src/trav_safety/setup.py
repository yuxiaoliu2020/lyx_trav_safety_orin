from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# 获取安装参数
setup_args = generate_distutils_setup(
    packages=['trav_safety'],
    package_dir={'': 'src'}
)

setup(**setup_args)