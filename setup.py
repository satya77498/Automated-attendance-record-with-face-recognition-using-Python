from cx_Freeze import setup, Executable
import sys,os
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

base = None

if sys.platform == 'win32':
    base = None


executables = [Executable("train.py", base="Win32GUI" if sys.platform == "win32" else base,
                          target_name="FaceAttendance.exe")]

packages = ["tkinter", "customtkinter", "cv2", "numpy", "PIL"]
options = {
    'build_exe': {
        'packages':packages,
        'include_files': [
            ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_default.xml"),
        ],
    },

}

setup(
    name = "ToolBox",
    options = options,
    version = "0.0.1",
    description = 'Vision ToolBox',
    executables = executables
)

#write python setup build