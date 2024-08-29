from setuptools import find_packages, setup

package_name = "cbf_controller"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "cbfpy @ git+https://github.com/danielpmorton/cbfpy.git",
        "numpy==1.26.4",
        "matplotlib",
        "pytest", # TODO move
        "pylint", # TODO move
        "black", # TODO move
    ],
    zip_safe=True,
    maintainer="Daniel Morton",
    maintainer_email="danielpmorton@gmail.com",
    description="Control Barrier Functions for SRC Drone Fencing Demo",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
