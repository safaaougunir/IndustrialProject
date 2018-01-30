#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui, QtCore


class Example(QtGui.QWidget):

    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def initUI(self):
        qbtn = QtGui.QPushButton('Upload your image here', self)
        #qbtn.clicked.connect(QtCore.QCoreApplication.instance().quit)
        #qbtn.clicked(self,self,self.upload_csv())$
        qbtn.clicked.connect(lambda: self.upload_csv())

        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 50)

        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Planmeric Your planning generator')
        self.show()
    def upload_csv(self):
        dialog = QtGui.QFileDialog()
        filename = dialog.getOpenFileName(None, 'Import Image ',"",  "jpg data files (*.jpg)")

        #fname = dialog.getOpenFileName(None, "Import CSV", "", "CSV data files (*.csv)")

def main():
    app = QtGui.QApplication(sys.argv)

    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()