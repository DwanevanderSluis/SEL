from setuptools import setup


def inc_version():
    fname = 'sellibrary/version.txt'
    try:
        file = open(fname, "r")
        version_str = file.readline()
    except FileNotFoundError as e :
        version_str = '0.0.0'
    v_list = version_str.split('.')
    v_list[-1] = str(int(v_list[-1])+1)
    version_str = ".".join(v_list)
    file = open(fname, "w")
    file.write(version_str)
    file.close()

    return version_str

version_str = inc_version()

setup(
    name='sellibrary',
    version=version_str,
    description='a pip-installable package example',
    license='MIT',
    packages=['sellibrary'],
    author='Dwane van der Sluis',
    author_email='ucabdv1@ucl.ac.uk',
    keywords=['example'], install_requires=['numpy', 'sklearn'],
    include_package_data=True
)