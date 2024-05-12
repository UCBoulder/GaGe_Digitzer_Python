# The point of this script is to create the necessary .py files in a PY
# subfolder of your QT project. You should be able to re-use this script for
# any QT project, just change the path variable below, and create a PY
# subfolder in your QT project directory!

import pyqt5ac

path = "./"
ioPaths = [
    [path + "*.ui", path + "PY/%%FILENAME%%.py"],
    [path + "*.qrc", path + "PY/%%FILENAME%%_rc.py"],
]

pyqt5ac.main(
    rccOptions="",
    uicOptions="--from-imports",
    force=False,
    config="",
    ioPaths=ioPaths,
    variables=None,
    initPackage=True,
)
