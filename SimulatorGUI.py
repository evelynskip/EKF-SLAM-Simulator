import sys
from PyQt5.QtWidgets import (QApplication, QWidget,QLabel,
QLineEdit,QPushButton,QMessageBox,QGridLayout,QGroupBox)
from PyQt5.QtCore import Qt,QRect
from PyQt5.QtGui import QPixmap,QFont
from qt_material import apply_stylesheet
import os
import SLAM_Algorithm
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import datetime


class EntryWindow(QWidget):
    def __init__(self):
        super().__init__()  # create default constructor for QWidget
        self.initializeUI()

    def initializeUI(self):
        extra = {
            # Button colors
            'danger': '#dc3545',
            'warning': '#ffc107',
            'success': '#17a2b8',

            # Font
            'font_family': 'Roboto',

        }
        self.setGeometry(500,250,1200,750)
        self.setWindowTitle('EKF-SLAM Simulator')
        apply_stylesheet(app, theme='light_blue.xml',invert_secondary=True, extra=extra)
        stylesheet = app.styleSheet()
        with open('custom.css') as file:
            app.setStyleSheet(stylesheet + file.read().format(**os.environ))
        self.displayWidgets()
        self.show()

    def displayWidgets(self):

        # Create name label and line edit widgets
        title_text=QLabel(self)
        title_text.setText("EKF-SLAM Simulator")
        title_text.move(70,70)
        title_text.setProperty('class', 'big_label')

        p1_label = QLabel("Interval:", self)
        p1_label.move(20, 150)
        self.p1_entry = QLineEdit(self)
        self.p1_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p1_entry.move(150, 150)
        self.p1_entry.resize(100, 20) # Change size of entry field
        p1_label_after = QLabel("s", self)
        p1_label_after.move(260, 150)

        p2_label = QLabel("Linear velocity:", self)
        p2_label.move(20, 200)
        self.p2_entry = QLineEdit(self)
        self.p2_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p2_entry.move(150, 200)
        self.p2_entry.resize(100, 20) # Change size of entry field 
        p2_label_after = QLabel("m/s", self)
        p2_label_after.move(260, 200)

        p3_label = QLabel("Angular velocity:", self)
        p3_label.move(20, 250)
        self.p3_entry = QLineEdit(self)
        self.p3_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p3_entry.move(150, 250)
        self.p3_entry.resize(100, 20) # Change size of entry field
        p2_label_after = QLabel("rad/s", self)
        p2_label_after.move(260, 250)

        p4_label = QLabel("Motion noise:", self)
        p4_label.move(20, 400)
        self.p4_entry = QLineEdit(self)
        self.p4_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p4_entry.move(150, 400)
        self.p4_entry.resize(100, 20) # Change size of entry field


        p5_label = QLabel("Measure. noise:", self)
        p5_label.move(20, 350)
        self.p5_entry = QLineEdit(self)
        self.p5_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p5_entry.move(150, 350)
        self.p5_entry.resize(100, 20) # Change size of entry field

        p6_label = QLabel("Observ. range:", self)
        p6_label.move(20, 300)
        self.p6_entry = QLineEdit(self)
        self.p6_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p6_entry.move(150, 300)
        self.p6_entry.resize(100, 20) # Change size of entry field
        p2_label_after = QLabel("m", self)
        p2_label_after.move(260, 300)

        groupBox = QGroupBox('EKF-SLAM Result',self)
        groupBox.setGeometry(QRect(450,30,700 ,660))


        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.gridlayout = QGridLayout(groupBox)  # 继承容器groupBox
        self.gridlayout.addWidget(self.canvas,0,1)
        self.ax=self.figure.add_axes([0.15,0.25,0.74,0.74])
        self.ax.set_xlabel("X/m")
        self.ax.set_ylabel("Y/m")
        self.ax1=self.figure.add_axes([0.15,0.07,0.74,0.10])
        self.ax1.set_xlabel("T/s")
        self.ax1.set_ylabel("Error/m")

        button_begin_dynamic = QPushButton('Begin_Dynamic',self)
        button_begin_dynamic.clicked.connect(self.begin_ekf_dynamic)
        button_begin_dynamic.move(60,600)

        button_begin_static = QPushButton('Begin_Static',self)
        button_begin_static.clicked.connect(self.begin_ekf_static)
        button_begin_static.move(210,600)

        button_clear = QPushButton('Set Default',self)
        button_clear.setProperty('class', 'danger')
        button_clear.clicked.connect(self.clear_ekf)
        button_clear.move(320,280)
   
        button_save = QPushButton('Save Image',self)
        button_save.setProperty('class', 'success')
        button_save.clicked.connect(self.save_image)
        button_save.move(60,550)

    def begin_ekf_dynamic(self):
        #v w rt qt
        DT=self.p1_entry.text()
        v=self.p2_entry.text()
        w=self.p3_entry.text()
        rt=self.p4_entry.text()
        qt=self.p5_entry.text()
        rng_max=self.p6_entry.text()
        SLAM_Algorithm.slam_function(self,0,DT,rt,qt,v,w,rng_max)

    
    def begin_ekf_static(self):
        DT=self.p1_entry.text()
        v=self.p2_entry.text()
        w=self.p3_entry.text()
        rt=self.p4_entry.text()
        qt=self.p5_entry.text()
        rng_max=self.p6_entry.text()
        SLAM_Algorithm.slam_function(self,1,DT,rt,qt,v,w,rng_max)
    
    def clear_ekf(self):
        self.p1_entry.setText('0.1')
        self.p2_entry.setText('2.0')
        self.p3_entry.setText('0.2')
        self.p4_entry.setText('0.01')
        self.p5_entry.setText('0.05')
        self.p6_entry.setText('15')
        


    def  save_image(self):
        time = datetime.datetime.now().strftime('%Y%m%d %H%M%S')
        filename='./Result/'+str(time)+'.jpg'
        plt.savefig(filename)

    def closeEvent(self, event):
        """
        Display a QMessageBox when asking the user if they want to quit the 
        program.
        """
        # Set up message box
        quit_msg = QMessageBox.question(self, "Quit Application?",
            "Are you sure you want to Quit?", QMessageBox.No | QMessageBox.Yes,
            QMessageBox.Yes)
        if quit_msg == QMessageBox.Yes:
            sys.exit()
            event.accept() # accept the event and close the application
        else:
            event.ignore() # ignore the close event
 

# run program
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EntryWindow()
    sys.exit(app.exec_())
    

    
