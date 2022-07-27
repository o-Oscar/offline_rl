import setuptools

requirements = ["matplotlib", "numpy", "torch"]

setuptools.setup(
    name="offline_rl",
    version="dev",
    author="Boutin Oscar",
    author_email="oscar.boutin.2019@polytechnique.org",
    install_requires=requirements,
)
