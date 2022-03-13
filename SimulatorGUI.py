import sys
from PyQt5.QtWidgets import (QApplication, QWidget,QLabel,
QLineEdit,QPushButton,QMessageBox,QGridLayout,QGroupBox)
from PyQt5.QtCore import Qt,QRect
from PyQt5.QtGui import QPixmap,QFont
from qt_material import apply_stylesheet
import os
import SLAM_Algorithm
# from EKF_SLAM import slam_function
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

        self.setGeometry(500,250,1200,700)
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

        p1_label = QLabel("DT:", self)
        p1_label.move(60, 150)
        self.p1_entry = QLineEdit(self)
        self.p1_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p1_entry.move(100, 150)
        self.p1_entry.resize(200, 20) # Change size of entry field

        p2_label = QLabel("Linear velocity:", self)
        p2_label.move(20, 200)
        self.p2_entry = QLineEdit(self)
        self.p2_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p2_entry.move(150, 200)
        self.p2_entry.resize(150, 20) # Change size of entry field 

        p3_label = QLabel("Angular velocity:", self)
        p3_label.move(20, 250)
        self.p3_entry = QLineEdit(self)
        self.p3_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p3_entry.move(150, 250)
        self.p3_entry.resize(150, 20) # Change size of entry field

        p4_label = QLabel("Motion noise:", self)
        p4_label.move(20, 300)
        self.p4_entry = QLineEdit(self)
        self.p4_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p4_entry.move(150, 300)
        self.p4_entry.resize(150, 20) # Change size of entry field


        p5_label = QLabel("Measure. noise:", self)
        p5_label.move(20, 350)
        self.p5_entry = QLineEdit(self)
        self.p5_entry.setAlignment(Qt.AlignLeft) # The default alignment is AlignLeft
        self.p5_entry.move(150, 350)
        self.p5_entry.resize(150, 20) # Change size of entry field

        groupBox = QGroupBox('EKF-SLAM Result',self)
        groupBox.setGeometry(QRect(450,70,700 ,550))


        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.gridlayout = QGridLayout(groupBox)  # 继承容器groupBox
        self.gridlayout.addWidget(self.canvas,0,1)
        self.ax=self.figure.add_axes([0.1,0.1,0.8,0.8])
        
        
        button_begin_dynamic = QPushButton('Begin_Dynamic',self)
        button_begin_dynamic.clicked.connect(self.begin_ekf_dynamic)
        button_begin_dynamic.move(60,550)#(550,70)

        button_begin_static = QPushButton('Begin_Static',self)
        button_begin_static.clicked.connect(self.begin_ekf_static)
        button_begin_static.move(210,550)#(550,70)

        button_clear = QPushButton('Set Default',self)
        button_clear.setProperty('class', 'danger')
        button_clear.clicked.connect(self.clear_ekf)
        button_clear.move(320,280)#(650,70)
   
        button_save = QPushButton('Save Image',self)
        button_save.setProperty('class', 'success')
        button_save.clicked.connect(self.save_image)
        button_save.move(60,500)#(650,70)

    def begin_ekf_dynamic(self):
        '''
        plt.clf()
        ax=self.figure.add_axes([0.1,0.1,0.8,0.8])
        t = np.arange(0.0, 5.0, 0.01)
        s = np.cos(2 * np.pi * t)
        ax.plot(t,s)
        self.canvas.draw()
        '''
        #v w rt qt
        DT=self.p1_entry.text()
        v=self.p2_entry.text()
        w=self.p3_entry.text()
        rt=self.p4_entry.text()
        qt=self.p5_entry.text()
        SLAM_Algorithm.slam_function(self,0,DT,rt,qt,v,w)

    
    def begin_ekf_static(self):
        DT=self.p1_entry.text()
        v=self.p2_entry.text()
        w=self.p3_entry.text()
        rt=self.p4_entry.text()
        qt=self.p5_entry.text()
        SLAM_Algorithm.slam_function(self,1,DT,rt,qt,v,w)
    
    def clear_ekf(self):
        self.p1_entry.setText('0.10')
        self.p2_entry.setText('2.0')
        self.p3_entry.setText('0.2')
        self.p4_entry.setText('0.1')
        self.p5_entry.setText('0.05')
        plt.clf()


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
    

    
