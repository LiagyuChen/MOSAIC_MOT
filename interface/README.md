## Get Started

### Run the Python Script
1. Install all the dependencies
```bash
git clone https://github.com/liagyuchen/LabelingInterface.git
cd LabelingInterface/
pip install -r requirements.txt
```
2. Run the script
```bash
python labeling_interface.py
```

### Compile to C executable using Nutika
1. Compile via Nutika command
```bash
nuitka labeling_interface.py --standalone --enable-plugin=tk-inter --enable-plugin=numpy --enable-plugin=torch --include-package=ultralytics --include-package=torch --include-package=torchvision --include-package=PIL --include-package=lap --include-package=cv2 --include-package=numpy  --include-data-files=.\\venv\\Lib\\site-packages\\ultralytics\\**\\*.yaml=ultralytics/ --include-data-dir=annotations=annotations --output-dir=build --nofollow-import-to=tkinter.test
```
2. Manually move all the files under `.\\venv\\Lib\\site-packages\\ultralytics\\cfg` to `.\\build\\labeling_interface.dist\\ultralytics\\cfg`

