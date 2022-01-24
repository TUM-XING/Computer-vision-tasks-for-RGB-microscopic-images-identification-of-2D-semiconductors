import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import test2



def main():
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = test2.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()