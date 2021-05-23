import sys
import site
site.addsitedir('/web_app/mypython/lib/python3.6/site-packages')
sys.path.insert(0, '/home/ubuntu/web_app/FLASK-YOLOV4')
from APIV4 import app as application